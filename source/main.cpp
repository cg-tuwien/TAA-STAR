#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include "helper_functions.hpp"
#include "taa.hpp"

/* TODO:
	- forward rendering
	- shadows?
	- rename mDrawCalls to something more appropriate (mMeshInfo?)
	- wtf? why does Sponza "crash" in debug mode due to alloc limit, but not in release mode?

	- history frame image format is RGBA8_SRGB, current frame is RGBA16_FLOAT.. is this ok???

	NOTES:
	- transparency pass without blending isn't bad either - needs larger alpha threshold ~0.5
*/

#define FORWARD_RENDERING 1

class wookiee : public gvk::invokee
{
	// Struct definition for data used as UBO across different pipelines, containing matrices and user input
	struct matrices_and_user_input
	{
		// view matrix as returned from quake_camera
		glm::mat4 mViewMatrix;
		// projection matrix as returned from quake_camera
		glm::mat4 mProjMatrix;
		// transformation matrix which tranforms to camera's position
		glm::mat4 mCamPos;
		// x = unused, y = normal mapping strength, z and w unused
		glm::vec4 mUserInput;
	};

	// Struct definition for data used as UBO across different pipelines, containing lightsource data
	struct lightsource_data
	{
		// x,y ... ambient light sources start and end indices; z,w ... directional light sources start and end indices
		glm::uvec4 mRangesAmbientDirectional;
		// x,y ... point light sources start and end indices; z,w ... spot light sources start and end indices
		glm::uvec4 mRangesPointSpot;
		// Contains all the data of all the active light sources
		std::array<gvk::lightsource_gpu_data, 128> mLightData;
	};

	// Constant data to be pushed per draw call that renders one or multiple meshes
	struct push_constant_data_per_drawcall {
		glm::mat4 mModelMatrix;
		int mMaterialIndex;
	};

	// Accumulated drawcall data for rendering the different meshes.
	// Includes an index which refers to an entry in the "list" of materials.
	struct drawcall_data
	{
		avk::buffer mIndexBuffer;
		avk::buffer mPositionsBuffer;
		avk::buffer mTexCoordsBuffer;
		avk::buffer mNormalsBuffer;
		avk::buffer mTangentsBuffer;
		avk::buffer mBitangentsBuffer;
		//push_constant_data_per_drawcall mPushConstants;
		std::vector<push_constant_data_per_drawcall> mPushConstantsVector;

		std::vector<uint32_t> mIndices;
		std::vector<glm::vec3> mPositions;
		std::vector<glm::vec2> mTexCoords;
		std::vector<glm::vec3> mNormals;
		std::vector<glm::vec3> mTangents;
		std::vector<glm::vec3> mBitangents;

		bool hasTransparency; // ac
	};

public: // v== cgb::cg_element overrides which will be invoked by the framework ==v
	static const uint32_t cConcurrentFrames = 3u;

	std::string mSceneFileName = "assets/sponza_with_plants_and_terrain.fscene";
	bool mDisableMip = false;
	bool mUseAlphaBlending = true;

	wookiee(avk::queue& aQueue)
		: mQueue{ &aQueue }
		, mAntiAliasing{ &aQueue }
	{}

	void prepare_matrices_ubo()
	{
		// Prepare a struct containing all the matrices.
		// This will be updated at the beginning of the render() callback
		//  (which is when we have the updated camera position available)
		mMatricesAndUserInput = { glm::mat4(), glm::mat4(), glm::mat4(), glm::vec4() };

		auto* wnd = gvk::context().main_window();
		auto fif = wnd->number_of_frames_in_flight();
		for (decltype(fif) i=0; i < fif; ++i) {
			mMatricesUserInputBuffer[i] = gvk::context().create_buffer(
				avk::memory_usage::host_coherent, {},
				avk::uniform_buffer_meta::create_from_data(mMatricesAndUserInput)
			);
			mMatricesUserInputBuffer[i]->fill(&mMatricesAndUserInput, 0, avk::sync::not_required());
		}
	}

	void update_matrices_and_user_input()
	{
		// Update the matrices in render() because here we can be sure that mQuakeCam's updates of the current frame are available:
		mMatricesAndUserInput = { mQuakeCam.view_matrix(), mQuakeCam.projection_matrix(), glm::translate(mQuakeCam.translation()), glm::vec4{ 0.f, mNormalMappingStrength, mUseLighting ? 1.f : 0.f, mAlphaThreshold } };
		const auto inFlightIndex = gvk::context().main_window()->in_flight_index_for_frame();
		mMatricesUserInputBuffer[inFlightIndex]->fill(&mMatricesAndUserInput, 0, avk::sync::not_required());
		// The cgb::sync::not_required() means that there will be no command buffer which the lifetime has to be handled of.
		// However, we have to ensure to properly sync memory dependency. In this application, this is ensured by the renderpass
		// dependency that is established between VK_SUBPASS_EXTERNAL and subpass 0.
	}

	void prepare_lightsources_ubo()
	{
		auto* wnd = gvk::context().main_window();
		auto fif = wnd->number_of_frames_in_flight();
		for (decltype(fif) i=0; i < fif; ++i) {
			mLightsourcesBuffer[i] = gvk::context().create_buffer(
				avk::memory_usage::host_coherent, {},
				avk::uniform_buffer_meta::create_from_size(sizeof(lightsource_data))
			);
		}
	}

	void update_lightsources()
	{
		// overwrite ambient and directional light
		auto lights = helpers::get_lights();
		int idxAmb = helpers::get_lightsource_type_begin_index(gvk::lightsource_type::ambient);
		int idxDir = helpers::get_lightsource_type_begin_index(gvk::lightsource_type::directional);
		lights[idxAmb] = gvk::lightsource::create_ambient(mAmbLight.col * mAmbLight.boost, "ambient light");
		lights[idxDir] = gvk::lightsource::create_directional(mDirLight.dir, mDirLight.intensity * mDirLight.boost, "directional light");

		lightsource_data updatedData {
			glm::uvec4{
				helpers::get_lightsource_type_begin_index(gvk::lightsource_type::ambient),
				helpers::get_lightsource_type_end_index(gvk::lightsource_type::ambient),
				helpers::get_lightsource_type_begin_index(gvk::lightsource_type::directional),
				helpers::get_lightsource_type_end_index(gvk::lightsource_type::directional)
			},
			glm::uvec4{
				helpers::get_lightsource_type_begin_index(gvk::lightsource_type::point),
				helpers::get_lightsource_type_end_index(gvk::lightsource_type::point),
				helpers::get_lightsource_type_begin_index(gvk::lightsource_type::spot),
				helpers::get_lightsource_type_end_index(gvk::lightsource_type::spot)
			},
			gvk::convert_for_gpu_usage<std::array<gvk::lightsource_gpu_data, 128>>(lights /* helpers::get_lights() */, mQuakeCam.view_matrix())
		};
		updatedData.mRangesPointSpot[1] = updatedData.mRangesPointSpot[0] + std::min(updatedData.mRangesPointSpot[1] - updatedData.mRangesPointSpot[0], static_cast<uint32_t>(mMaxPointLightCount));
		updatedData.mRangesPointSpot[3] = updatedData.mRangesPointSpot[2] + std::min(updatedData.mRangesPointSpot[3] - updatedData.mRangesPointSpot[2], static_cast<uint32_t>(mMaxSpotLightCount));

		const auto inFlightIndex = gvk::context().main_window()->in_flight_index_for_frame();
		mLightsourcesBuffer[inFlightIndex]->fill(&updatedData, 0, avk::sync::not_required());
	}

	void prepare_framebuffers_and_post_process_images()
	{
		using namespace avk;
		using namespace gvk;

		auto* wnd = gvk::context().main_window();

		// Before compiling the actual framebuffer, create its image-attachments:
		const auto wndRes = wnd->resolution();
		// Define the formats of our image-attachments:
		auto attachmentFormats = make_array<vk::Format>(
			vk::Format::eR16G16B16A16Sfloat,
			vk::Format::eD32Sfloat,
			vk::Format::eR32G32B32A32Sfloat,
			vk::Format::eR32Uint
		);

		auto fif = wnd->number_of_frames_in_flight();
		for (decltype(fif) i=0; i < fif; ++i) {
			auto colorAttachment = context().create_image(wndRes.x, wndRes.y, attachmentFormats[0], 1, memory_usage::device, image_usage::general_color_attachment | image_usage::input_attachment);
			colorAttachment->set_target_layout(vk::ImageLayout::eShaderReadOnlyOptimal); // <-- because afterwards, we are going to read from it when applying the post processing effects
			auto depthAttachment = context().create_image(wndRes.x, wndRes.y, attachmentFormats[1], 1, memory_usage::device, image_usage::general_depth_stencil_attachment | image_usage::input_attachment);
			depthAttachment->set_target_layout(vk::ImageLayout::eShaderReadOnlyOptimal); // <-- because afterwards, we are going to read from it when applying the post processing effects
#if (!FORWARD_RENDERING)
			auto uvNrmAttachment = context().create_image(wndRes.x, wndRes.y, attachmentFormats[2], 1, memory_usage::device, image_usage::general_color_attachment | image_usage::input_attachment);
			uvNrmAttachment->set_target_layout(vk::ImageLayout::eShaderReadOnlyOptimal); // <-- because afterwards, we are going to read from it whene applying the post processing effects
			auto matIdAttachment = context().create_image(wndRes.x, wndRes.y, attachmentFormats[3], 1, memory_usage::device, image_usage::general_color_attachment | image_usage::input_attachment);
			matIdAttachment->set_target_layout(vk::ImageLayout::eShaderReadOnlyOptimal); // <-- because afterwards, we are going to read from it when applying the post processing effects
#endif

			// Before we are attaching the images to a framebuffer, we have to "wrap" them with an image view:
			auto colorAttachmentView = context().create_image_view(std::move(colorAttachment));
			colorAttachmentView.enable_shared_ownership(); // We are using this attachment in both, the mFramebuffer and the mSkyboxFramebuffer
			auto depthAttachmentView = context().create_depth_image_view(std::move(depthAttachment));
#if (!FORWARD_RENDERING)
			auto uvNrmAttachmentView = context().create_image_view(std::move(uvNrmAttachment));
			auto matIdAttachmentView = context().create_image_view(std::move(matIdAttachment));
#endif

			// Create the renderpass only once, ...
			if (0 == i) {
				// Both, `cgb::framebuffer_t::create` and `cgb::graphics_pipeline_for` support attachments being passed directly.
				// However, we can also pass a `cgb::renderpass` instance instead of the attachments. In fact, if attachments
				// are passed directly, a renderpass is created under the hood anyways. In this case, let's just re-use the same
				// renderpass for the `mFramebuffer` and the pipelines that we're going to create in `prepare_deferred_shading_pipelines()`.
				//
				// Note: For rendering with a pipeline into a framebuffer, Vulkan does not demand that the associated renderpass
				//       INSTANCES are the same. Vulkan allows different renderpass instances, but their LAYOUTS have to be COMPATIBLE,
				//       which means that the number of attachments, their formats, and also their subpasses are the same. They are
				//       allowed to differ in some other parameters like load/store operations.
				mRenderpass = context().create_renderpass({
						// For each attachment, describe exactly how we intend to use it:
						//   1st parameter: read some parameters like `image_format` and `image_usage` from the image views
						//   2nd parameter: What shall be done before rendering starts? Shall the image be cleared or maybe its contents be preserved (a.k.a. "load")?
						//   3rd parameter: How are we going to use this attachment in the subpasses?
						//   4th parameter: What shall be done when the last subpass has finished? => Store the image contents?
#if FORWARD_RENDERING
						attachment::declare_for(colorAttachmentView, on_load::load,         color(0) -> color(0),			on_store::store),
						attachment::declare_for(depthAttachmentView, on_load::clear, depth_stencil() -> depth_stencil(),	on_store::store),
#else
						attachment::declare_for(colorAttachmentView, on_load::load,         unused() -> color(0), on_store::store),
						attachment::declare_for(depthAttachmentView, on_load::clear, depth_stencil() -> input(0), on_store::store),
						attachment::declare_for(uvNrmAttachmentView, on_load::clear,        color(0) -> input(1), on_store::store),
						attachment::declare_for(matIdAttachmentView, on_load::clear,        color(1) -> input(2), on_store::store)
#endif
					},
					[](avk::renderpass_sync& aRpSync){
						// Synchronize with everything that comes BEFORE:
						if (aRpSync.is_external_pre_sync()) {
							// This pipeline must wait on the skybox pipeline's color attachment writes to become available, before writing to (the same!) color attachment:
							aRpSync.mSourceStage                    = avk::pipeline_stage::color_attachment_output;
							aRpSync.mDestinationStage               = avk::pipeline_stage::color_attachment_output;
							// It's the same memory access for both, that needs to be synchronized:
							aRpSync.mSourceMemoryDependency         = avk::memory_access::color_attachment_write_access;
							aRpSync.mDestinationMemoryDependency    = avk::memory_access::color_attachment_write_access;
						}
						// Synchronize with everything that comes AFTER:
						if (aRpSync.is_external_post_sync()) {
							// After this, definitely a compute shader will be invoked. That compute shader needs to wait for color attachments to be made available:
							aRpSync.mSourceStage                    = avk::pipeline_stage::color_attachment_output;
							aRpSync.mSourceMemoryDependency         = avk::memory_access::color_attachment_write_access;
							// ...however mayyyybe, we'll have a TRANSFER command afterwards (e.g. Blit) => make sure to synchronize both cases:
							aRpSync.mDestinationStage               = avk::pipeline_stage::compute_shader                       | avk::pipeline_stage::transfer;
							aRpSync.mDestinationMemoryDependency    = avk::memory_access::shader_buffers_and_images_read_access | avk::memory_access::transfer_read_access;
						}
					}
				);
				mRenderpass.enable_shared_ownership(); // We're not going to have only one instance of this renderpass => turn it into a shared pointer (internally) so we can re-use it.
			}

			// ... but one framebuffer per frame in flight
			mFramebuffer[i] = context().create_framebuffer(
				mRenderpass,
				          colorAttachmentView,
				std::move(depthAttachmentView)
#if (!FORWARD_RENDERING)
				,
				std::move(uvNrmAttachmentView),
				std::move(matIdAttachmentView)
#endif
			);
			mFramebuffer[i]->initialize_attachments(sync::wait_idle(true));

			// Create the framebuffer for the skybox:
			mSkyboxFramebuffer[i] = context().create_framebuffer(
				{ attachment::declare_for(colorAttachmentView, on_load::dont_care, color(0), on_store::store) }, 
				colorAttachmentView
			);
			mSkyboxFramebuffer[i]->initialize_attachments(avk::sync::wait_idle(true));
		}
	}

	void prepare_skybox()
	{
		using namespace avk;
		using namespace gvk;

		auto* wnd = gvk::context().main_window();
		
		// Load a sphere model for drawing the skybox:
		auto sphere = model_t::load_from_file("assets/sphere.obj");
		std::tie(mSphereVertexBuffer, mSphereIndexBuffer) = create_vertex_and_index_buffers( make_models_and_meshes_selection(sphere, 0), {}, sync::wait_idle(true) );

		// Create the graphics pipeline to be used for drawing the skybox:
		mSkyboxPipeline = context().create_graphics_pipeline_for(
			// Shaders to be used with this pipeline:
			vertex_shader("shaders/sky_gradient.vert"),
			fragment_shader("shaders/sky_gradient.frag"),
			// Declare the vertex input to the shaders:
			from_buffer_binding(0) -> stream_per_vertex<glm::vec3>() -> to_location(0),
			context().create_renderpass({
					attachment::declare(vk::Format::eR16G16B16A16Sfloat, on_load::dont_care, color(0), on_store::store)
				},
				[](avk::renderpass_sync& aRpSync){
					// Synchronize with everything that comes BEFORE:
					if (aRpSync.is_external_pre_sync()) {
						// This renderpass is invoked BEFORE anything else => no need to wait for anything
						aRpSync.mSourceStage                    = avk::pipeline_stage::top_of_pipe;
						aRpSync.mSourceMemoryDependency         = {};
						// But the color attachment must be ready to be written into
						aRpSync.mDestinationStage               = avk::pipeline_stage::color_attachment_output;
						aRpSync.mDestinationMemoryDependency    = avk::memory_access::color_attachment_write_access;
					}
					// Synchronize with everything that comes AFTER:
					if (aRpSync.is_external_post_sync()) {
						// The next pipeline must wait before this pipeline has finished writing its color attachment output
						aRpSync.mSourceStage                    = avk::pipeline_stage::color_attachment_output;
						aRpSync.mDestinationStage               = avk::pipeline_stage::color_attachment_output;
						// It's the same memory access for both, that needs to be synchronized:
						aRpSync.mSourceMemoryDependency         = avk::memory_access::color_attachment_write_access;
						aRpSync.mDestinationMemoryDependency    = avk::memory_access::color_attachment_write_access;
					}
				}
			),
			// Further config for the pipeline:
			cfg::culling_mode::disabled,	// No backface culling required
			cfg::depth_test::disabled(),	// No depth test required
			cfg::depth_write::disabled(),	// Don't write depth values
			cfg::viewport_depth_scissors_config::from_framebuffer(wnd->backbuffer_at_index(0)), // Set to the dimensions of the main window
			descriptor_binding(0, 0, mMatricesUserInputBuffer[0])
		);

		auto& commandPool = gvk::context().get_command_pool_for_reusable_command_buffers(*mQueue);
		
		// Create a command buffer and record the commands for rendering the skybox into it:
		// (We will record the drawing commands once, and use/"replay" it every frame.)
		auto fif = wnd->number_of_frames_in_flight();
		for (decltype(fif) i=0; i < fif; ++i) {
			mSkyboxCommandBuffer[i] = commandPool->alloc_command_buffer();	
			mSkyboxCommandBuffer[i]->begin_recording(); // Start recording commands into the command buffer
			helpers::record_timing_interval_start(mSkyboxCommandBuffer[i]->handle(), fmt::format("mSkyboxCommandBuffer{} time", i));
			mSkyboxCommandBuffer[i]->bind_pipeline(mSkyboxPipeline);
			mSkyboxCommandBuffer[i]->begin_render_pass_for_framebuffer( // Start the renderpass defined for the attachments (in fact, the renderpass is created FROM the attachments, see renderpass_t::create)
				mSkyboxPipeline->get_renderpass(),                   // <-- We'll use the pipeline's renderpass, where we have defined load/store operations.
				mSkyboxFramebuffer[i]
			);
			mSkyboxCommandBuffer[i]->bind_descriptors(mSkyboxPipeline->layout(), mDescriptorCache.get_or_create_descriptor_sets({  // Bind the descriptors which describe resources used by shaders.
				descriptor_binding(0, 0, mMatricesUserInputBuffer[i])                                  // In this case, we have one uniform buffer as resource (we have also declared that during mSkyboxPipeline creation).
			}));
			mSkyboxCommandBuffer[i]->draw_indexed(*mSphereIndexBuffer, *mSphereVertexBuffer); // Record the draw call
			mSkyboxCommandBuffer[i]->end_render_pass();
			helpers::record_timing_interval_end(mSkyboxCommandBuffer[i]->handle(), fmt::format("mSkyboxCommandBuffer{} time", i));
			mSkyboxCommandBuffer[i]->end_recording(); // Done recording. We're not going to modify this command buffer anymore.
		}
	}

	// ac: output info about the scene structure
	void print_scene_debug_info(gvk::orca_scene_t& aScene, bool withMeshNames = false)
	{
		auto &models = aScene.models();
		std::cout << "Scene debug info" << std::endl;
		std::cout << "#models " << models.size() << std::endl;
		for (int iModel = 0; iModel < models.size(); iModel++) {
			auto assimpScene = models[iModel].mLoadedModel->handle();
			std::cout << "Model #" << iModel << ", (" << assimpScene->mNumMeshes << " meshes):" << std::endl;
			int meshCount = 0;
			lukeMeshWalker(assimpScene, withMeshNames, assimpScene->mRootNode, meshCount);
			std::cout << "Total # meshes in scene graph:" << meshCount << std::endl;
		}
	}

	void lukeMeshWalker(const aiScene *aiscene, bool withMeshNames, aiNode *node, int &meshCount) {
		std::string fullName = node->mName.C_Str();
		aiNode *n = node->mParent;
		while (n) {
			fullName.insert(0, "/");
			fullName.insert(0, n->mName.C_Str());
			n = n ->mParent;
		}
		std::string meshNames;
		std::cout << fullName << ":\t" << node->mNumMeshes << " meshes:";
		for (size_t i = 0; i < node->mNumMeshes; i++) {
			std::cout << " " << node->mMeshes[i];
			if (withMeshNames) {
				if (meshNames.length()) meshNames.append(",");
				meshNames.append("\"");
				meshNames.append(aiscene->mMeshes[node->mMeshes[i]]->mName.C_Str());
				meshNames.append("\"");
			}
		}

		if (withMeshNames) {
			std::cout << " (" << meshNames << ")" << std::endl;
		} else {
			std::cout << std::endl;
		}

		meshCount += node->mNumMeshes;
		for (size_t iChild = 0; iChild < node->mNumChildren; iChild++)
			lukeMeshWalker(aiscene, withMeshNames, node->mChildren[iChild], meshCount);
	}

	uint32_t colToUint32(glm::vec4 col) {
		glm::ivec4 c(col * 255.f);
		return ((c.r & 0xff) << 24) | ((c.g & 0xff) << 16) | ((c.b & 0xff) << 8) | (c.a & 0xff);
	}

	std::string baseName(std::string path) {
		return path.substr(path.find_last_of("/\\") + 1);
	}

	// ac: print material properties
	void print_material_debug_info(gvk::orca_scene_t& aScene) {
		auto distinctMaterialsOrca = aScene.distinct_material_configs_for_all_models(true);
		std::cout << "Material debug info" << std::endl;
		int cnt = 0;
		for (const auto& pair : distinctMaterialsOrca) {
			cnt++;
			auto mc = pair.first;
			std::cout << "Material " << cnt << "/" << distinctMaterialsOrca.size() << ": \"" << mc.mName << "\":" << std::endl;
			if (mc.mShadingModel.length())				{ std::cout << " Shading model:     " << mc.mShadingModel << std::endl;						}
			if (mc.mTransparentColor != glm::vec4(0))	{ std::cout << " Transparent color: "; printf("%.8x\n", colToUint32(mc.mTransparentColor));	}
			if (mc.mEmissiveColor    != glm::vec4(0))	{ std::cout << " Emissive color:    "; printf("%.8x\n", colToUint32(mc.mEmissiveColor));	}
			if (mc.mDiffuseTex.length())				{ std::cout << " Diffuse tex:       " << baseName(mc.mDiffuseTex) << std::endl;				}
			if (mc.mEmissiveTex.length())				{ std::cout << " Emissive tex:      " << baseName(mc.mEmissiveTex) << std::endl;			}
			if (mc.mOpacity != 1.f)						{ std::cout << " Opacity:           " << mc.mOpacity << std::endl;							}
			if (mc.mTwosided)							{ std::cout << " Twosided:          TRUE" << std::endl;										}
		}
	}

	// Create a vector of transformation matrices for each instance of a given mesh id inside a model
	std::vector<glm::mat4> get_mesh_instance_transforms(const gvk::model &m, int meshId, const glm::mat4 &baseTransform) {
		std::vector<glm::mat4> transforms;
		collect_mesh_transforms_from_node(meshId, m->handle()->mRootNode, baseTransform, transforms);
		return transforms;
	}

	void collect_mesh_transforms_from_node(int meshId, aiNode * node, const glm::mat4 &accTransform, std::vector<glm::mat4> &transforms) {
		glm::mat4 newAccTransform = accTransform * aiMat4_to_glmMat4(node->mTransformation);

		for (size_t i = 0; i < node->mNumMeshes; i++) {
			if (node->mMeshes[i] == meshId) {
				transforms.push_back(newAccTransform);
			}
		}
		for (size_t iChild = 0; iChild < node->mNumChildren; iChild++)
			collect_mesh_transforms_from_node(meshId, node->mChildren[iChild], newAccTransform, transforms);
	}

	static glm::mat4 aiMat4_to_glmMat4(const aiMatrix4x4 &ai) {
		glm::mat4 g;
		g[0][0] = ai[0][0]; g[0][1] = ai[1][0]; g[0][2] = ai[2][0]; g[0][3] = ai[3][0];
		g[1][0] = ai[0][1]; g[1][1] = ai[1][1]; g[1][2] = ai[2][1]; g[1][3] = ai[3][1];
		g[2][0] = ai[0][2]; g[2][1] = ai[1][2]; g[2][2] = ai[2][2]; g[2][3] = ai[3][2];
		g[3][0] = ai[0][3]; g[3][1] = ai[1][3]; g[3][2] = ai[2][3]; g[3][3] = ai[3][3];
		return g;
	}

	bool has_material_transparency(const gvk::material_config &mat) {
		// In Emerald Square v4, all materials with transparent parts are named "*.DoubleSided"
		return (std::string::npos != mat.mName.find(".DoubleSided"));
	}

	void load_and_prepare_scene() // up to the point where all draw call data and material data has been assembled
	{
		double t0 = glfwGetTime();
		std::cout << "Loading scene..." << std::endl;

		// Load a scene (in ORCA format) from file:
		auto scene = gvk::orca_scene_t::load_from_file(mSceneFileName, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace /*| aiProcess_FlipUVs */);

		double tLoad = glfwGetTime();

		// print scene graph
		//print_scene_debug_info(scene);
		//print_material_debug_info(scene);

		// Change the materials of "terrain" and "debris", enable tessellation for them, and set displacement scaling:
		helpers::set_terrain_material_config(scene);
		
		auto bufferUsageFlags = vk::BufferUsageFlags{};
#ifdef RTX_ON
		assert(cgb::settings::gEnableBufferDeviceAddress);
		bufferUsageFlags |= vk::BufferUsageFlagBits::eShaderDeviceAddressKHR;
#endif

		std::cout << "Parsing scene\r"; std::cout.flush();


		int counter = 0, counterlimit = 0; // 200;

		// Get all the different materials from the whole scene:
		auto distinctMaterialsOrca = scene->distinct_material_configs_for_all_models();
		std::vector<gvk::material_config> distinctMaterialConfigs;
		for (const auto& pair : distinctMaterialsOrca) {
			// Also gather the material configs along the way since we'll need to transfer them to the GPU as well
			// (which we will do right after this for loop that gathers the draw call data).
			const int materialIndex = static_cast<int>(distinctMaterialConfigs.size());
			distinctMaterialConfigs.push_back(pair.first);
			assert (static_cast<size_t>(materialIndex + 1) == distinctMaterialConfigs.size());

			bool materialHasTransparency = has_material_transparency(pair.first);

			for (const auto& modelAndMeshIndices : pair.second) {
				// Gather the model reference and the mesh indices of the same material in a vector:
				auto& modelData = scene->model_at_index(modelAndMeshIndices.mModelIndex);
				std::vector<std::tuple<std::reference_wrapper<const gvk::model_t>, std::vector<size_t>>> modelRefAndMeshIndices = { std::make_tuple(std::cref(modelData.mLoadedModel), modelAndMeshIndices.mMeshIndices) };
				helpers::exclude_a_curtain(modelRefAndMeshIndices);
				if (modelRefAndMeshIndices.empty()) {
					continue;
				}

				//// Create a draw call for all those gathered meshes, once per ORCA-instance:
				//for (size_t i = 0; i < modelData.mInstances.size(); ++i) {
				//	auto [vertices, indices] = gvk::get_vertices_and_indices(modelRefAndMeshIndices);
				//	auto texCoords = gvk::get_2d_texture_coordinates(modelRefAndMeshIndices, 0);
				//	auto normals = gvk::get_normals(modelRefAndMeshIndices);
				//	auto tangents = gvk::get_tangents(modelRefAndMeshIndices);
				//	auto bitangents = gvk::get_bitangents(modelRefAndMeshIndices);
				//	auto& ref = mDrawCalls.emplace_back(drawcall_data {
				//		// Create all the GPU buffers, but don't fill yet:
				//		gvk::context().create_buffer(avk::memory_usage::device, bufferUsageFlags, avk::index_buffer_meta::create_from_data(indices)),
				//		gvk::context().create_buffer(avk::memory_usage::device, bufferUsageFlags, avk::vertex_buffer_meta::create_from_data(vertices).describe_only_member(vertices[0], avk::content_description::position)),
				//		gvk::context().create_buffer(avk::memory_usage::device, bufferUsageFlags, avk::vertex_buffer_meta::create_from_data(texCoords)),
				//		gvk::context().create_buffer(avk::memory_usage::device, bufferUsageFlags, avk::vertex_buffer_meta::create_from_data(normals)),
				//		gvk::context().create_buffer(avk::memory_usage::device, bufferUsageFlags, avk::vertex_buffer_meta::create_from_data(tangents)),
				//		gvk::context().create_buffer(avk::memory_usage::device, bufferUsageFlags, avk::vertex_buffer_meta::create_from_data(bitangents)),
				//		push_constant_data_per_drawcall {
				//			gvk::matrix_from_transforms(modelData.mInstances[i].mTranslation, glm::quat(modelData.mInstances[i].mRotation), modelData.mInstances[i].mScaling),
				//			materialIndex // Assign material at the given index to this draw call
				//		}
				//	});
				//	ref.mIndices = std::move(indices);
				//	ref.mPositions = std::move(vertices);
				//	ref.mTexCoords = std::move(texCoords);
				//	ref.mNormals = std::move(normals);
				//	ref.mTangents = std::move(tangents);
				//	ref.mBitangents = std::move(bitangents);
				//}

				// Create a draw call for all those gathered meshes, once per ORCA-instance:
				// ac: scene graph fix, quick & dirty for now
				// - use one drawcall per mesh (or more drawcalls if the mesh appears in multiple nodes); don't combine meshes
				// TODO: combine meshes that are unique
				// TODO: use instanced drawing?
				for (size_t i = 0; i < modelData.mInstances.size(); ++i) {
					for (auto& pair : modelRefAndMeshIndices) {
						for (auto meshIndex : std::get<std::vector<size_t>>(pair)) {

							counter++;
							std::cout << "Parsing scene " << counter << "\r"; std::cout.flush();

							std::vector<size_t> tmpMeshIndexVector = { meshIndex };
							std::vector<std::tuple<std::reference_wrapper<const gvk::model_t>, std::vector<size_t>>> singleMesh_modelRefAndMeshIndices = { std::make_tuple(std::cref(modelData.mLoadedModel), tmpMeshIndexVector) };

							// get all the common (per-mesh) properties
							auto [vertices, indices] = gvk::get_vertices_and_indices(singleMesh_modelRefAndMeshIndices);
							auto texCoords = gvk::get_2d_texture_coordinates(singleMesh_modelRefAndMeshIndices, 0);
							auto normals = gvk::get_normals(singleMesh_modelRefAndMeshIndices);
							auto tangents = gvk::get_tangents(singleMesh_modelRefAndMeshIndices);
							auto bitangents = gvk::get_bitangents(singleMesh_modelRefAndMeshIndices);

							// collect all the instances of the mesh (it may appear in multiple nodes, thus using different transforms)
							auto modelBaseTransform = gvk::matrix_from_transforms(modelData.mInstances[i].mTranslation, glm::quat(modelData.mInstances[i].mRotation), modelData.mInstances[i].mScaling);
							auto transforms = get_mesh_instance_transforms(modelData.mLoadedModel, meshIndex, modelBaseTransform);
							//std::cout << "Mesh " << meshIndex << ": " << transforms.size() << " transforms" << std::endl;

							// build push constants (per mesh-instance) vector
							std::vector<push_constant_data_per_drawcall> pcvec;
							for (auto tform : transforms) pcvec.push_back(push_constant_data_per_drawcall{ tform, materialIndex });

							auto& ref = mDrawCalls.emplace_back(drawcall_data{
								// Create all the GPU buffers, but don't fill yet:
								gvk::context().create_buffer(avk::memory_usage::device, bufferUsageFlags, avk::index_buffer_meta::create_from_data(indices)),
								gvk::context().create_buffer(avk::memory_usage::device, bufferUsageFlags, avk::vertex_buffer_meta::create_from_data(vertices).describe_only_member(vertices[0], avk::content_description::position)),
								gvk::context().create_buffer(avk::memory_usage::device, bufferUsageFlags, avk::vertex_buffer_meta::create_from_data(texCoords)),
								gvk::context().create_buffer(avk::memory_usage::device, bufferUsageFlags, avk::vertex_buffer_meta::create_from_data(normals)),
								gvk::context().create_buffer(avk::memory_usage::device, bufferUsageFlags, avk::vertex_buffer_meta::create_from_data(tangents)),
								gvk::context().create_buffer(avk::memory_usage::device, bufferUsageFlags, avk::vertex_buffer_meta::create_from_data(bitangents)),
								pcvec
								});
							ref.mIndices = std::move(indices);
							ref.mPositions = std::move(vertices);
							ref.mTexCoords = std::move(texCoords);
							ref.mNormals = std::move(normals);
							ref.mTangents = std::move(tangents);
							ref.mBitangents = std::move(bitangents);

							ref.hasTransparency = materialHasTransparency;
						}
					}
				}
			}
			if (counter > counterlimit && counterlimit > 0) break;
		}
		std::cout << std::endl;

		// Convert the material configs (that were gathered above) into a GPU-compatible format:
		// "GPU-compatible format" in this sense means that we'll get two things out of the call to `convert_for_gpu_usage`:
		//   1) Material data in a properly aligned format, suitable for being uploaded into a GPU buffer (but not uploaded into a buffer yet!)
		//   2) Image samplers (which contain images and samplers) of all the used textures, already uploaded to the GPU.
		std::tie(mMaterialData, mImageSamplers) = gvk::convert_for_gpu_usage(
			distinctMaterialConfigs, false,
			mDisableMip ? avk::image_usage::general_image : avk::image_usage::general_texture,
			[](){ return avk::to_filter_mode(gvk::context().physical_device().getProperties().limits.maxSamplerAnisotropy, true); }(), // set to max. anisotropy
			avk::border_handling_mode::repeat,
			avk::sync::wait_idle(true)
		);

		// Create a GPU buffer that will get filled with the material data:
		mMaterialBuffer = gvk::context().create_buffer(
			avk::memory_usage::device, {},
			avk::storage_buffer_meta::create_from_data(mMaterialData)
		);

		// ac: get the dir light source from the scene file
		auto dirLights = scene->directional_lights();
		if (dirLights.size()) {
			mDirLight.dir       = dirLights[0].mDirection;
			mDirLight.intensity = dirLights[0].mIntensity;
			mDirLight.boost     = 1.f;
		}


		double tParse = glfwGetTime();
		printf("Loading took %.1f sec, parsing took %.1f sec, total = %.1f sec\n", tLoad-t0, tParse-tLoad, tParse-t0);
	}

	void upload_materials_and_vertex_data_to_gpu()
	{
		// All of the following are submitted to the same queue (due to cgb::device_queue_selection_strategy::prefer_everything_on_single_queue)
		// That also means that this is the same queue which is used for graphics rendering.
		// Furthermore, this means that it is sufficient to establish a memory barrier after the last call to cgb::fill. 
		for (auto& dc : mDrawCalls) {
			                                                       // Take care of the command buffer's lifetime, but do not establish a barrier before or after the command
			dc.mIndexBuffer     ->fill(dc.mIndices.data(),    0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
			dc.mPositionsBuffer ->fill(dc.mPositions.data(),  0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
			dc.mTexCoordsBuffer ->fill(dc.mTexCoords.data(),  0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
			dc.mNormalsBuffer   ->fill(dc.mNormals.data(),    0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
			dc.mTangentsBuffer  ->fill(dc.mTangents.data(),   0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
			dc.mBitangentsBuffer->fill(dc.mBitangents.data(), 0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
		}

		// This is the last one of our TRANSFER commands. 
		mMaterialBuffer->fill(mMaterialData.data(), 0, avk::sync::with_barriers(
			[this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, // Also in this case, take care of the command buffer's lifetime
			{}, // No need to establish a barrier before it, but... 
			[](avk::command_buffer_t& cb, avk::pipeline_stage srcStage, std::optional<avk::write_memory_access> srcAccess){ // ... establish a barrier AFTER the last TRANSFER command
				cb.establish_global_memory_barrier_rw(
					srcStage,   /* -> */ avk::pipeline_stage::all_graphics, // All graphics stages must wait for the above buffers to be completely transfered
					srcAccess,  /* -> */ avk::memory_access::any_graphics_read_access // Transfered memory must be made visible to any kind of of graphics pipeline read access types.
				);
			}));
	}

	void prepare_deferred_shading_pipelines()
	{
		using namespace avk;
		using namespace gvk;

		mPipelineFirstPass = context().create_graphics_pipeline_for(
			// Specify which shaders the pipeline consists of (type is inferred from the extension):
			"shaders/transform_and_pass_on.vert",
			"shaders/blinnphong_and_normal_mapping.frag",
			// The next lines define the format and location of the vertex shader inputs:
			// (The dummy values (like glm::vec3) tell the pipeline the format of the respective input)
			from_buffer_binding(0) -> stream_per_vertex<glm::vec3>() -> to_location(0),		// <-- corresponds to vertex shader's aPosition
			from_buffer_binding(1) -> stream_per_vertex<glm::vec2>() -> to_location(1),		// <-- corresponds to vertex shader's aTexCoords
			from_buffer_binding(2) -> stream_per_vertex<glm::vec3>() -> to_location(2),		// <-- corresponds to vertex shader's aNormal
			from_buffer_binding(3) -> stream_per_vertex<glm::vec3>() -> to_location(3),		// <-- corresponds to vertex shader's aTangent
			from_buffer_binding(4) -> stream_per_vertex<glm::vec3>() -> to_location(4),		// <-- corresponds to vertex shader's aBitangent
			// Some further settings:
			cfg::front_face::define_front_faces_to_be_counter_clockwise(),
			cfg::viewport_depth_scissors_config::from_framebuffer(mFramebuffer[0]),
			mRenderpass, 0u, // Use this pipeline for subpass #0 of the specified renderpass
			//
			// The following define additional data which we'll pass to the pipeline:
			//   We'll pass two matrices to our vertex shader via push constants:
			push_constant_binding_data { shader_type::all, 0, sizeof(push_constant_data_per_drawcall) }, // We also have to declare that we're going to submit push constants
			descriptor_binding(0, 0, mMaterialBuffer),	// As far as used resources are concerned, we need the materials buffer (type: vk::DescriptorType::eStorageBuffer),
			descriptor_binding(0, 1, mImageSamplers),		// multiple images along with their sampler (array of vk::DescriptorType::eCombinedImageSampler),
			descriptor_binding(1, 0, mMatricesUserInputBuffer[0]),
			descriptor_binding(1, 1, mLightsourcesBuffer[0])
		);

		mPipelineLightingPass = context().create_graphics_pipeline_for(
			"shaders/lighting_pass.vert",
			"shaders/lighting_pass.frag",
			from_buffer_binding(0) -> stream_per_vertex(&helpers::quad_vertex::mPosition)          -> to_location(0),
			from_buffer_binding(0) -> stream_per_vertex(&helpers::quad_vertex::mTextureCoordinate) -> to_location(1),
			cfg::front_face::define_front_faces_to_be_clockwise(),
			cfg::viewport_depth_scissors_config::from_framebuffer(mFramebuffer[0]),
			cfg::culling_mode::disabled,
			cfg::depth_test::disabled(),
			mRenderpass, 1u, // <-- Use this pipeline for subpass #1 of the specified renderpass
			//
			// Adding the push constants (same size as mPipelineFirstPass) here might seem totally unneccessary,
			// and actually it is from a shader-centric point of view. However, by establishing a "compatible layout"
			// between the two different pipelines for push constants and descriptor sets 0 and 1, we can re-use
			// the descriptors across the pipelines and do not have to bind them again for each pipeline.
			// (see record_command_buffer_for_models() when drawing with the mPipelineLightingPass pipeline)
			push_constant_binding_data { shader_type::all, 0, sizeof(push_constant_data_per_drawcall) },
			// Further bindings which contribute to the pipeline layout:
			//   Descriptor sets 0 and 1 are exactly the same as in mPipelineFirstPass
			descriptor_binding(0, 0, mMaterialBuffer),
			descriptor_binding(0, 1, mImageSamplers),
			descriptor_binding(1, 0, mMatricesUserInputBuffer[0]),
			descriptor_binding(1, 1, mLightsourcesBuffer[0]),
			descriptor_binding(2, 0, mFramebuffer[0]->image_view_at(1)->as_input_attachment(), shader_type::fragment),
			descriptor_binding(2, 1, mFramebuffer[0]->image_view_at(2)->as_input_attachment(), shader_type::fragment),
			descriptor_binding(2, 2, mFramebuffer[0]->image_view_at(3)->as_input_attachment(), shader_type::fragment)
		);
	}

	void prepare_forward_rendering_pipelines()
	{
		using namespace avk;
		using namespace gvk;

		const uint32_t specConstId_transparentPass = 1u; // corresponds to shader: layout(constant_id = 1) const uint transparentPass

		mPipelineFwdOpaque = context().create_graphics_pipeline_for(
			// Specify which shaders the pipeline consists of (type is inferred from the extension):
			"shaders/transform_and_pass_on.vert",
			fragment_shader("shaders/fwd_geometry.frag").set_specialization_constant(specConstId_transparentPass, uint32_t{ 0 }), // 0 = opaque pass
			// The next lines define the format and location of the vertex shader inputs:
			// (The dummy values (like glm::vec3) tell the pipeline the format of the respective input)
			from_buffer_binding(0) -> stream_per_vertex<glm::vec3>() -> to_location(0),		// <-- corresponds to vertex shader's aPosition
			from_buffer_binding(1) -> stream_per_vertex<glm::vec2>() -> to_location(1),		// <-- corresponds to vertex shader's aTexCoords
			from_buffer_binding(2) -> stream_per_vertex<glm::vec3>() -> to_location(2),		// <-- corresponds to vertex shader's aNormal
			from_buffer_binding(3) -> stream_per_vertex<glm::vec3>() -> to_location(3),		// <-- corresponds to vertex shader's aTangent
			from_buffer_binding(4) -> stream_per_vertex<glm::vec3>() -> to_location(4),		// <-- corresponds to vertex shader's aBitangent
			// Some further settings:
			cfg::front_face::define_front_faces_to_be_counter_clockwise(),
			cfg::viewport_depth_scissors_config::from_framebuffer(mFramebuffer[0]),
			mRenderpass, 0u, // Use this pipeline for subpass #0 of the specified renderpass
			//
			// The following define additional data which we'll pass to the pipeline:
			//   We'll pass two matrices to our vertex shader via push constants:
			push_constant_binding_data { shader_type::all, 0, sizeof(push_constant_data_per_drawcall) }, // We also have to declare that we're going to submit push constants
			descriptor_binding(0, 0, mMaterialBuffer),	// As far as used resources are concerned, we need the materials buffer (type: vk::DescriptorType::eStorageBuffer),
			descriptor_binding(0, 1, mImageSamplers),		// multiple images along with their sampler (array of vk::DescriptorType::eCombinedImageSampler),
			descriptor_binding(1, 0, mMatricesUserInputBuffer[0]),
			descriptor_binding(1, 1, mLightsourcesBuffer[0])
		);

		// this is almost the same, except for the specialization constant, alpha blending (and backface culling? depth write?)
		mPipelineFwdTransparent = context().create_graphics_pipeline_for(
			"shaders/transform_and_pass_on.vert",
			fragment_shader("shaders/fwd_geometry.frag").set_specialization_constant(specConstId_transparentPass, uint32_t{ 1 }), // 1 = transparent pass

			cfg::color_blending_config::enable_alpha_blending_for_all_attachments(),
			cfg::culling_mode::disabled,
			// cfg::depth_write::disabled(), // would need back-to-front sorting, also a problem for TAA... so leave it on (and render only stuff with alpha >= threshold)
			// cfg::depth_test::disabled(),  // not good, definitely needs sorting

			// The next lines define the format and location of the vertex shader inputs:
			// (The dummy values (like glm::vec3) tell the pipeline the format of the respective input)
			from_buffer_binding(0) -> stream_per_vertex<glm::vec3>() -> to_location(0),		// <-- corresponds to vertex shader's aPosition
			from_buffer_binding(1) -> stream_per_vertex<glm::vec2>() -> to_location(1),		// <-- corresponds to vertex shader's aTexCoords
			from_buffer_binding(2) -> stream_per_vertex<glm::vec3>() -> to_location(2),		// <-- corresponds to vertex shader's aNormal
			from_buffer_binding(3) -> stream_per_vertex<glm::vec3>() -> to_location(3),		// <-- corresponds to vertex shader's aTangent
			from_buffer_binding(4) -> stream_per_vertex<glm::vec3>() -> to_location(4),		// <-- corresponds to vertex shader's aBitangent
			// Some further settings:
			cfg::front_face::define_front_faces_to_be_clockwise(),
			cfg::viewport_depth_scissors_config::from_framebuffer(mFramebuffer[0]),
			mRenderpass, 1u, // <-- Use this pipeline for subpass #1 of the specified renderpass
			//
			push_constant_binding_data { shader_type::all, 0, sizeof(push_constant_data_per_drawcall) }, // We also have to declare that we're going to submit push constants
			descriptor_binding(0, 0, mMaterialBuffer),	// As far as used resources are concerned, we need the materials buffer (type: vk::DescriptorType::eStorageBuffer),
			descriptor_binding(0, 1, mImageSamplers),		// multiple images along with their sampler (array of vk::DescriptorType::eCombinedImageSampler),
			descriptor_binding(1, 0, mMatricesUserInputBuffer[0]),
			descriptor_binding(1, 1, mLightsourcesBuffer[0])
		);

		// alternative pipeline - render transparency with simple alpha test only, no blending
		mPipelineFwdTransparentNoBlend = context().create_graphics_pipeline_for(
			"shaders/transform_and_pass_on.vert",
			fragment_shader("shaders/fwd_geometry.frag").set_specialization_constant(specConstId_transparentPass, uint32_t{ 1 }), // 1 = transparent pass

			cfg::culling_mode::disabled,
			// cfg::depth_write::disabled(), // would need back-to-front sorting, also a problem for TAA... so leave it on (and render only stuff with alpha >= threshold)
			// cfg::depth_test::disabled(),  // not good, definitely needs sorting

			// The next lines define the format and location of the vertex shader inputs:
			// (The dummy values (like glm::vec3) tell the pipeline the format of the respective input)
			from_buffer_binding(0) -> stream_per_vertex<glm::vec3>() -> to_location(0),		// <-- corresponds to vertex shader's aPosition
			from_buffer_binding(1) -> stream_per_vertex<glm::vec2>() -> to_location(1),		// <-- corresponds to vertex shader's aTexCoords
			from_buffer_binding(2) -> stream_per_vertex<glm::vec3>() -> to_location(2),		// <-- corresponds to vertex shader's aNormal
			from_buffer_binding(3) -> stream_per_vertex<glm::vec3>() -> to_location(3),		// <-- corresponds to vertex shader's aTangent
			from_buffer_binding(4) -> stream_per_vertex<glm::vec3>() -> to_location(4),		// <-- corresponds to vertex shader's aBitangent
																							// Some further settings:
			cfg::front_face::define_front_faces_to_be_clockwise(),
			cfg::viewport_depth_scissors_config::from_framebuffer(mFramebuffer[0]),
			mRenderpass, 1u, // <-- Use this pipeline for subpass #1 of the specified renderpass
							 //
			push_constant_binding_data { shader_type::all, 0, sizeof(push_constant_data_per_drawcall) }, // We also have to declare that we're going to submit push constants
			descriptor_binding(0, 0, mMaterialBuffer),	// As far as used resources are concerned, we need the materials buffer (type: vk::DescriptorType::eStorageBuffer),
			descriptor_binding(0, 1, mImageSamplers),		// multiple images along with their sampler (array of vk::DescriptorType::eCombinedImageSampler),
			descriptor_binding(1, 0, mMatricesUserInputBuffer[0]),
			descriptor_binding(1, 1, mLightsourcesBuffer[0])
		);

	}

	// Record actual draw calls for all the drawcall-data that we have
	// gathered in load_and_prepare_scene() into a command buffer
	void record_command_buffer_for_models()
	{
		using namespace avk;
		using namespace gvk;

		auto* wnd = gvk::context().main_window();
		//auto& commandPool = context().get_command_pool_for_reusable_command_buffers(*mQueue);
		auto& commandPool = context().get_command_pool_for_resettable_command_buffers(*mQueue); // ac: we may need to re-record

		mAlphaBlendingActive = mUseAlphaBlending;

#if FORWARD_RENDERING
		const auto &firstPipe  = mPipelineFwdOpaque;
		const auto &secondPipe = mUseAlphaBlending ? mPipelineFwdTransparent : mPipelineFwdTransparentNoBlend;
#else
		const auto &firstPipe  = mPipelineFirstPass;
		const auto &secondPipe = mPipelineLightingPass;
#endif

		auto fif = wnd->number_of_frames_in_flight();
		for (decltype(fif) i=0; i < fif; ++i) {
			if (!mDidAllocCommandBuffers) mModelsCommandBuffer[i] = commandPool->alloc_command_buffer();
			mModelsCommandBuffer[i]->begin_recording();
			helpers::record_timing_interval_start(mModelsCommandBuffer[i]->handle(), fmt::format("mModelsCommandBuffer{} time", i));

			// Bind the descriptors for descriptor sets 0 and 1 before starting to render with a pipeline
			mModelsCommandBuffer[i]->bind_descriptors(firstPipe->layout(), mDescriptorCache.get_or_create_descriptor_sets({ // They must match the pipeline's layout (per set!) exactly.
				descriptor_binding(0, 0, mMaterialBuffer),
				descriptor_binding(0, 1, mImageSamplers),
				descriptor_binding(1, 0, mMatricesUserInputBuffer[i]),
				descriptor_binding(1, 1, mLightsourcesBuffer[i])
			}));

			// Draw using our pipeline for the first pass (Initially this is the only
			//   pass. After task 2 has been implemented, this is the G-Buffer pass):
			mModelsCommandBuffer[i]->bind_pipeline(firstPipe);
			mModelsCommandBuffer[i]->begin_render_pass_for_framebuffer(firstPipe->get_renderpass(), mFramebuffer[i]);

			// Record all the draw calls into this command buffer:
			for (auto& drawCall : mDrawCalls) {
				// In forward rendering, only render opaque geometry in the first pass
#if FORWARD_RENDERING
				if (drawCall.hasTransparency) continue;
#endif
				// Issue a separate drawcall for each transform stored in drawCall.mPushConstantsVector ; ac: TODO: make that an instanced draw ?
				for (auto pushc : drawCall.mPushConstantsVector) {

					// Set model/material-specific data for this draw call in the form of push constants:
					mModelsCommandBuffer[i]->push_constants(
						firstPipe->layout(),	// Push constants must match the pipeline's layout
						pushc					// Push the actual data
					);

					// Bind the vertex input buffers in the right order (corresponding to the layout
					// specifiers in the vertex shader) and issue the actual draw call:
					mModelsCommandBuffer[i]->draw_indexed(
						*drawCall.mIndexBuffer,
						*drawCall.mPositionsBuffer,
						*drawCall.mTexCoordsBuffer,
						*drawCall.mNormalsBuffer,
						*drawCall.mTangentsBuffer,
						*drawCall.mBitangentsBuffer
					);
				}
			}

#if FORWARD_RENDERING
			mModelsCommandBuffer[i]->next_subpass();
			mModelsCommandBuffer[i]->bind_pipeline(secondPipe);
			for (auto& drawCall : mDrawCalls) {
				// render only transparent geometry
				if (!drawCall.hasTransparency) continue;
				for (auto pushc : drawCall.mPushConstantsVector) {
					mModelsCommandBuffer[i]->push_constants(
						firstPipe->layout(),	// Push constants must match the pipeline's layout
						pushc					// Push the actual data
					);
					mModelsCommandBuffer[i]->draw_indexed(
						*drawCall.mIndexBuffer,
						*drawCall.mPositionsBuffer,
						*drawCall.mTexCoordsBuffer,
						*drawCall.mNormalsBuffer,
						*drawCall.mTangentsBuffer,
						*drawCall.mBitangentsBuffer
					);
				}
			}
#else
			// Move on to next subpass, synchronizing all data to be written to memory,
			// and to be made visible to the next subpass, which uses it as input.
			mModelsCommandBuffer[i]->next_subpass();
			mModelsCommandBuffer[i]->bind_pipeline(secondPipe);
			mModelsCommandBuffer[i]->bind_descriptors(secondPipe->layout(), mDescriptorCache.get_or_create_descriptor_sets({ 
				descriptor_binding(0, 0, mMaterialBuffer),
				descriptor_binding(0, 1, mImageSamplers),
				descriptor_binding(1, 0, mMatricesUserInputBuffer[i]),
				descriptor_binding(1, 1, mLightsourcesBuffer[i]),
				descriptor_binding(2, 0, mFramebuffer[i]->image_view_at(1)->as_input_attachment(), shader_type::fragment),
				descriptor_binding(2, 1, mFramebuffer[i]->image_view_at(2)->as_input_attachment(), shader_type::fragment),
				descriptor_binding(2, 2, mFramebuffer[i]->image_view_at(3)->as_input_attachment(), shader_type::fragment)
			}));

			const auto& [quadVertices, quadIndices] = helpers::get_quad_vertices_and_indices();
			mModelsCommandBuffer[i]->draw_indexed(quadIndices, quadVertices);
#endif

			helpers::record_timing_interval_end(mModelsCommandBuffer[i]->handle(), fmt::format("mModelsCommandBuffer{} time", i));
			mModelsCommandBuffer[i]->end_render_pass();
			mModelsCommandBuffer[i]->end_recording();
		}

		mDidAllocCommandBuffers = true;
	}

	void setup_ui_callback()
	{
		using namespace avk;
		using namespace gvk;
		
		auto imguiManager = current_composition()->element_by_type<imgui_manager>();
		if(nullptr != imguiManager) {
			imguiManager->add_callback([this](){

				static auto smplr = context().create_sampler(filter_mode::bilinear, border_handling_mode::clamp_to_edge);
				static auto texIdsAndDescriptions = [&](){

					// Gather all the framebuffer attachments to display them
					std::vector<std::tuple<std::optional<ImTextureID>, std::string>> v;
					const auto n = mFramebuffer[0]->image_views().size();
					for (size_t i = 0; i < n; ++i) {
						if (mFramebuffer[0]->image_view_at(i)->get_image().config().samples != vk::SampleCountFlagBits::e1) {
							LOG_INFO(fmt::format("Excluding framebuffer attachment #{} from the UI because it has a sample count != 1. Wouldn't be displayed properly, sorry.", i));
							v.emplace_back(std::optional<ImTextureID>{}, fmt::format("Not displaying attachment #{}", i));
						}
						else {
							if (!is_norm_format(mFramebuffer[0]->image_view_at(i)->get_image().config().format) && !is_float_format(mFramebuffer[0]->image_view_at(i)->get_image().config().format)) {
								LOG_INFO(fmt::format("Excluding framebuffer attachment #{} from the UI because it has format that can not be sampled with a (floating point-type) sampler2D.", i));
								v.emplace_back(std::optional<ImTextureID>{}, fmt::format("Not displaying attachment #{}", i));
							}
							else {
								v.emplace_back(
									ImGui_ImplVulkan_AddTexture(smplr->handle(), mFramebuffer[0]->image_view_at(i)->handle(), static_cast<VkImageLayout>(vk::ImageLayout::eShaderReadOnlyOptimal)),
									fmt::format("Attachment #{}", i)
								);
							}
						}
					}
					return v;

				}();

				auto inFlightIndex = context().main_window()->in_flight_index_for_frame();

				ImGui::Begin("Info & Settings");
				ImGui::SetWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
				ImGui::SetWindowSize(ImVec2(250.0f, 700.0f), ImGuiCond_FirstUseEver);
				ImGui::Text("%.3f ms/frame", 1000.0f / ImGui::GetIO().Framerate);
				ImGui::Text("%.3f ms/mSkyboxCommandBuffer", helpers::get_timing_interval_in_ms(fmt::format("mSkyboxCommandBuffer{} time", inFlightIndex)));
				ImGui::Text("%.3f ms/mModelsCommandBuffer", helpers::get_timing_interval_in_ms(fmt::format("mModelsCommandBuffer{} time", inFlightIndex)));
				ImGui::Text("%.3f ms/Anti Aliasing",        mAntiAliasing.duration());
				ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);

				static std::vector<float> accum; // accumulate (then average) 10 frames
				accum.push_back(ImGui::GetIO().Framerate);
				static std::vector<float> values;
				if (accum.size() == 10) {
					values.push_back(std::accumulate(std::begin(accum), std::end(accum), 0) / 10.0f);
					accum.clear();
				}
				if (values.size() > 90) { // Display up to 90(*10) history frames
					values.erase(values.begin());
				}
				ImGui::PlotLines("FPS", values.data(), values.size(), 0, nullptr, 0.0f, FLT_MAX, ImVec2(0.0f, 150.0f));

				ImGui::TextColored(ImVec4(0.f, .6f, .8f, 1.f), "[F1]: Toggle input-mode");
				ImGui::TextColored(ImVec4(0.f, .6f, .8f, 1.f), " (UI vs. scene navigation)");
				
				ImGui::SliderInt("max point lights", &mMaxPointLightCount, 0, 98);
				ImGui::SliderInt("max spot lights", &mMaxSpotLightCount, 0, 11);

				ImGui::ColorEdit3("dir col", &mDirLight.intensity.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel); ImGui::SameLine();
				ImGui::InputFloat3("dir light", &mDirLight.dir.x);
				ImGui::SliderFloat("dir boost", &mDirLight.boost, 0.f, 1.f);

				ImGui::ColorEdit3("amb col", &mAmbLight.col.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel); ImGui::SameLine();
				ImGui::SliderFloat("amb boost", &mAmbLight.boost, 0.f, 1.f);

				ImGui::Checkbox("use lights", &mUseLighting);

				ImGui::SliderFloat("alpha thres", &mAlphaThreshold, 0.f, 1.f, "%.3f", 2.f);
				ImGui::Checkbox("alpha blending", &mUseAlphaBlending);

				ImGui::SliderFloat("Normal Mapping Strength", &mNormalMappingStrength, 0.0f, 1.0f);

				ImGui::Separator();

				for (auto& tpl : texIdsAndDescriptions) {
					auto texId = std::get<std::optional<ImTextureID>>(tpl);
					auto description = std::get<std::string>(tpl);
					if (texId.has_value()) {
						ImGui::Image(texId.value(), ImVec2(192, 108)); ImGui::SameLine();
					}
					ImGui::Text(description.c_str());
				}

				ImGui::End();
			});
		}
		else {
			LOG_WARNING("No component of type cgb::imgui_manager found => could not install ImGui callback.");
		}
	}

	void initialize() override
	{
		using namespace avk;
		using namespace gvk;

		auto* wnd = context().main_window();
		
		// Create a descriptor cache that helps us to conveniently create descriptor sets:
		mDescriptorCache = gvk::context().create_descriptor_cache();
		
		prepare_matrices_ubo();
		prepare_lightsources_ubo();
		prepare_framebuffers_and_post_process_images();
		prepare_skybox();
		load_and_prepare_scene();
#if FORWARD_RENDERING
		prepare_forward_rendering_pipelines();
#else
		prepare_deferred_shading_pipelines();
#endif
		record_command_buffer_for_models();

		// Add the camera to the composition (and let it handle the updates)
		mQuakeCam.set_translation({ 0.0f, 1.0f, 0.0f });
		//mQuakeCam.set_perspective_projection(glm::radians(60.0f), context().main_window()->aspect_ratio(), 0.1f, 500.0f);
		mQuakeCam.set_perspective_projection(glm::radians(60.0f), context().main_window()->aspect_ratio(), 0.1f, 5000.0f); // ac testing far plane
		mOriginalProjMat = mQuakeCam.projection_matrix();
		current_composition()->add_element(mQuakeCam);

		setup_ui_callback();

		upload_materials_and_vertex_data_to_gpu();

		std::array<image_view*, cConcurrentFrames> srcDepthImages;
//		std::array<image_view*, cConcurrentFrames> srcUvNrmImages;
//		std::array<image_view*, cConcurrentFrames> srcMatIdImages;
		std::array<image_view, cConcurrentFrames> srcColorImages;
		auto fif = wnd->number_of_frames_in_flight();
		for (decltype(fif) i = 0; i < fif; ++i) {
			srcDepthImages[i] = &mFramebuffer[i]->image_view_at(1);
//			srcUvNrmImages[i] = &mFramebuffer[i]->image_view_at(2);
//			srcMatIdImages[i] = &mFramebuffer[i]->image_view_at(3);
			srcColorImages[i] = mFramebuffer[i]->image_view_at(0);
		}

		mAntiAliasing.set_source_image_views(srcColorImages, srcDepthImages);
		current_composition()->add_element(mAntiAliasing);
	}

	void update() override
	{
		const auto inFlightIndex = gvk::context().main_window()->in_flight_index_for_frame();
		if (inFlightIndex == 1 + cConcurrentFrames) {
			mStoredCommandBuffers.clear();
		}
		
		// Let Temporal Anti-Aliasing modify the camera's projection matrix:
		auto* mainWnd = gvk::context().main_window();
		auto modifiedProjMat = mAntiAliasing.get_jittered_projection_matrix(mOriginalProjMat, mainWnd->current_frame());
		mAntiAliasing.save_history_proj_matrix(mOriginalProjMat, mainWnd->current_frame());

		mQuakeCam.set_projection_matrix(modifiedProjMat);
		
		if (gvk::input().key_pressed(gvk::key_code::escape)) {
			// Stop the current composition:
			gvk::current_composition()->stop();
		}

		// [F1]: Toggle quake cam's modes
		if (gvk::input().key_pressed(gvk::key_code::f1)) {
			auto imguiManager = gvk::current_composition()->element_by_type<gvk::imgui_manager>();
			if (mQuakeCam.is_enabled()) {
				mQuakeCam.disable();
				if (nullptr != imguiManager) { imguiManager->enable_user_interaction(true); }
			}
			else {
				mQuakeCam.enable();
				if (nullptr != imguiManager) { imguiManager->enable_user_interaction(false); }
			}
		}

		// re-record command buffers if necessary
		if (mUseAlphaBlending != mAlphaBlendingActive) {
			gvk::context().device().waitIdle();
			record_command_buffer_for_models();
		}

		// helpers::animate_lights(helpers::get_lights(), gvk::time().absolute_time());
	}

	void render() override
	{
		auto mainWnd = gvk::context().main_window();
		update_matrices_and_user_input();
		update_lightsources();

		auto inFlightIndex = mainWnd->in_flight_index_for_frame();
		mQueue->submit(mSkyboxCommandBuffer[inFlightIndex], std::optional<std::reference_wrapper<avk::semaphore_t>> {});
		mQueue->submit(mModelsCommandBuffer[inFlightIndex], std::optional<std::reference_wrapper<avk::semaphore_t>> {});

		// anti_asiasing::render() will be invoked after this
	}

	void finalize() override
	{
		helpers::clean_up_timing_resources();
		gvk::current_composition()->remove_element(mQuakeCam);
	}

private: // v== Member variables ==v

	avk::queue* mQueue;
	avk::descriptor_cache mDescriptorCache;
	
	std::vector<avk::command_buffer> mStoredCommandBuffers;
	
	// Our camera for navigating the scene
	glm::mat4 mOriginalProjMat;
	gvk::quake_camera mQuakeCam;

	int mMaxPointLightCount = 0; // std::numeric_limits<int>::max();
	int mMaxSpotLightCount  = 0; // std::numeric_limits<int>::max();
	
	// Data for the matrices UBO
	float mNormalMappingStrength = 0.7f;
	matrices_and_user_input mMatricesAndUserInput;
	std::array<avk::buffer, cConcurrentFrames> mMatricesUserInputBuffer;

	std::array<avk::buffer, cConcurrentFrames> mLightsourcesBuffer;

	// Framebuffer to render into
	avk::renderpass mRenderpass;
	std::array<avk::framebuffer, cConcurrentFrames> mFramebuffer;
	std::array<avk::framebuffer, cConcurrentFrames> mSkyboxFramebuffer;

	// Data for rendering the skybox
	avk::buffer mSphereVertexBuffer;
	avk::buffer mSphereIndexBuffer;
	avk::graphics_pipeline mSkyboxPipeline;
	std::array<avk::command_buffer, cConcurrentFrames> mSkyboxCommandBuffer;

	// Data for rendering the 3D models
	std::vector<gvk::material_gpu_data> mMaterialData;
	avk::buffer mMaterialBuffer;
	std::vector<avk::image_sampler> mImageSamplers;
	std::vector<drawcall_data> mDrawCalls;
	std::array<avk::command_buffer, cConcurrentFrames> mModelsCommandBuffer;

	// Different pipelines used for (deferred) shading:
	avk::graphics_pipeline mPipelineFirstPass;
	avk::graphics_pipeline mPipelineLightingPass;

	// Pipelines for forward rendering:
	avk::graphics_pipeline mPipelineFwdOpaque;
	avk::graphics_pipeline mPipelineFwdTransparent;
	avk::graphics_pipeline mPipelineFwdTransparentNoBlend;

	// The elements to handle the post processing effects:
	taa<cConcurrentFrames> mAntiAliasing;

	bool mUseLighting = true; // to en/disable light processing in shader
	struct { glm::vec3 dir, intensity; float boost; } mDirLight = { {1.f,1.f,1.f}, { 1.f,1.f,1.f }, 1.f }; // this is overwritten with the dir light from the .fscene file
	struct { glm::vec3 col; float boost; } mAmbLight = { {1.f, 1.f, 1.f}, 0.1f };

	float mAlphaThreshold = 0.001f; // alpha threshold for rendering transparent parts
	bool mAlphaBlendingActive;
	bool mDidAllocCommandBuffers = false;
};

int main(int argc, char **argv) // <== Starting point ==
{
	try {
		// Parse command line
		// first parameter starting without dash is scene filename
		bool badCmd = false;
		bool disableValidation = false;
		bool forceValidation = false;
		bool disableMip = false;
		bool disableAlphaBlending = false;
		std::string sceneFileName = "";
		for (int i = 1; i < argc; i++) {
			if (0 == strncmp("-", argv[i], 1)) {
				if (0 == _stricmp(argv[i], "-novalidation")) {
					disableValidation = true;
					LOG_INFO("Validation layers disabled via command line parameter.");
				} else if (0 == _stricmp(argv[i], "-validation")) {
					forceValidation = true;
					LOG_INFO("Validation layers enforced via command line parameter.");
				} else if (0 == _stricmp(argv[i], "-nomip")) {
					disableMip = true;
					LOG_INFO("Mip-mapping disabled via command line parameter.");
				} else if (0 == _stricmp(argv[i], "-noblend")) {
					disableAlphaBlending = true;
					LOG_INFO("Alpha-blending disabled via command line parameter.");
				} else {
					badCmd = true;
					break;
				}
			} else {
				if (sceneFileName.length()) {
					badCmd = true;
					break;
				}
				sceneFileName = argv[i];
			}
		}
		if (badCmd) {
			printf("Usage: %s [-novalidation] [-validation] [-nomip] [orca scene file path]\n", argv[0]);
			return EXIT_FAILURE;
		}


		// Create a window and open it
		auto mainWnd = gvk::context().create_window("TAA-STAR");
		mainWnd->set_resolution({ 1920, 1080 });
		mainWnd->set_additional_back_buffer_attachments({ 
			avk::attachment::declare(vk::Format::eD32Sfloat, avk::on_load::clear, avk::depth_stencil(), avk::on_store::dont_care)
		});
		mainWnd->set_presentaton_mode(gvk::presentation_mode::mailbox);
		mainWnd->set_number_of_presentable_images(3u);
		mainWnd->set_number_of_concurrent_frames(wookiee::cConcurrentFrames);
		mainWnd->request_srgb_framebuffer(true);
		mainWnd->open();

		auto& singleQueue = gvk::context().create_queue({}, avk::queue_selection_preference::versatile_queue, mainWnd);
		mainWnd->add_queue_family_ownership(singleQueue);
		mainWnd->set_present_queue(singleQueue);
		
		// Create an instance of our main avk::element which contains all the functionality:
		auto chewbacca = wookiee(singleQueue);
		// Create another element for drawing the UI with ImGui
		auto ui = gvk::imgui_manager(singleQueue);

		// ac: disable validation layers via command line (renderdoc crashes when they are enabled....)
		gvk::validation_layers val_layers = {};
		val_layers.enable_in_release_mode(forceValidation); // note: in release, this doesn't enable the debug callback, but val.errors are dumped to the console
		if (disableValidation) val_layers.mLayers.clear();

		// set scene file name and other command line params
		if (sceneFileName.length()) chewbacca.mSceneFileName = sceneFileName;
		chewbacca.mDisableMip = disableMip;
		chewbacca.mUseAlphaBlending = !disableAlphaBlending;


		// GO:
		gvk::start(
			gvk::application_name("TAA-STAR"),
			mainWnd,
			chewbacca,
			ui,
			val_layers
		);
	}
	catch (gvk::logic_error&) {}
	catch (gvk::runtime_error&) {}
	catch (avk::logic_error&) {}
	catch (avk::runtime_error&) {}
}
