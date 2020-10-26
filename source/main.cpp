#include "debug_helper.hpp"

#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include "helper_functions.hpp"

#include "taa.hpp"

#include "splines.hpp"

#include <string>

// use forward rendering? (if 0: use deferred shading)
#define FORWARD_RENDERING 1

// use the gvk updater for shader hot reloading and window resizing ?
#define USE_GVK_UPDATER 1

// record (model) command buffer in render() instead of prerecording? (as workaround to allow shader hot reloading); note this is quite a bit slower (~ +2ms for Emerald Square)
#define RECORD_CMDBUFFER_IN_RENDER 1

/* TODO:
	still problems with slow-mo when capturing frames - use /frame instead of /sec when capturing for now!

	- strange "seam" in smooth sphere and soccer ball -> check if that is the model or some other problem!
	  => it's the model (soccerball, has holes too), and for the sphere normal mapping was the problem (no uv, no tangents)

	- moving objs with more than one mesh/material

	- when using deferred shading - any point in using a compute shader for the lighting pass ?

	- motion vectors problems, like: object coming into view from behind an obstacle. tags?

	- is there any point to keep using 2 render-subpasses in forward rendering?

	- recheck lighting, esp. w.r.t. twosided materials

	- fix changed lighting flags in deferred shader

	- shadows?
	- rename mDrawCalls to something more appropriate (mMeshInfo?)

	- need different alpha thresholds for blending/not blending

	NOTES:
	- transparency pass without blending isn't bad either - needs larger alpha threshold ~0.5
*/


/*
	DrawIndexedIndirect scheme:

	create one big vertex and index buffer (for scenery only? - tbd) [and matching buffers for other vertex-attributes: texCoord,normals,tangents,bitangents]

	meshgroup ~= what is now called drawcall_data -> all the instances for a given meshid (differing only in their modelmatrix)

	build an array of VkDrawIndexedIndirectCommand (size = #meshgroups)
	for each meshgroup mg:
		instance count = #instances in mg
		first index = start index of mg in huge index buffer
		vertex offset = start vertex of mg in huge vertex buffer	(or keep 0 and adjust indices accordingly)
		keep first instance = 0

	create a materialIndexBuffer[#mg]	(NO alternative: stream it in via instance attributes? - is a waste of memory, need to replicate per-mg-value for each mesh-instance; there are no per-draw-attributes unfortunately)
	create a perInstanceAttributesBuffer[# total instances] (for now holds only model matrix) (could stream this in via instance attribs instead)
	create a attribStartIndexBuffer[#mg] -> holds index for perInstanceAttributesBuffer for the first instance of the mg (if not streaming attrib buffer)
	(for motion vectors on static geo: attribs of last frame == same as this frame, so don't need anything extra)

	for now we don't stream the per-instance attributes

	example perInstanceAttributesBuffer layout (if mesh0 has 2 instances, mesh1 has 1 and mesh2 has 3):
		perInstanceAttributesBuffer { mesh0_inst0, mesh0_inst1, mesh1_inst0, mesh2_inst0, mesh2_inst1, mesh2_inst2, mesh3_inst0, ... }
		attribStartIndexBuffer		{ 0, 2, 3, 6, ... }

	draw call/pipeline:
		per-vertex-attributes:	 index, vertex, normals, tan, bitan, ...
		per-instance-attributes: attribStartIndex (when not streaming instace attribs), or the instance-attributes directly
		buffer materialIndexBuffer[]

	in shader:
		gl_DrawId        := mg being drawn
		gl_InstanceIndex := instance of mg being drawn

		materialIdx = materialIndexBuffer[gl_DrawId]

		// via buffers
		attribBase = attribStartIndexBuffer[gl_DrawId]
		modelMatrix = perInstanceAttributesBuffer[attribBase + gl_InstanceIndex].modelmatrix

		// via streaming attribs
		modelMatrix = in_perInstanceAttributes_modelmatrix


	test status OK: also tested with scenes with multiple orca-models, multiple orca-instances
*/

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

		glm::mat4 mPrevFrameProjViewMatrix;
		glm::mat4 mMovingObjectModelMatrix;
		glm::mat4 mPrevFrameMovingObjectModelMatrix;
		glm::vec4 mJitterCurrentPrev;
		int  mActiveMovingObjectMaterialIdx;
		int  mPrevFrameActiveMovingObjectMaterialIdx;
		float mLodBias;
		float pad1;
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

	// Constant data to be pushed per draw call that renders one or multiple meshes (not used with the DrawIndexedIndirect variant)
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

	// push constants for DrawIndexedIndirect
	struct push_constant_data_for_dii {
		int mDrawIdOffset;
	};

	struct MeshgroupPerInstanceData {
		glm::mat4 modelMatrix;
	};
	struct Meshgroup {
		uint32_t numIndices;
		uint32_t numVertices;
		uint32_t baseIndex;		// indexbuffer-index  corresponding to the first index of this meshid
		uint32_t baseVertex;	// vertexbuffer-index corresponding to the first index of this meshid (?)
		std::vector<MeshgroupPerInstanceData> perInstanceData;
		int      materialIndex;
		bool     hasTransparency;

		// these are mainly for debugging:
		uint32_t orcaModelId;
		uint32_t orcaMeshId;
	};

	struct CameraState { char name[80];  glm::vec3 t; glm::quat r; };	// ugly char[80] for easier ImGui access...

	std::vector<CameraState> mCameraPresets = {			// NOTE: literal quat constructor = {w,x,y,z} 
		{ "Start" },	// t,r filled in from code
		{ "ES street flicker",       {-18.6704f, 3.43254f, 17.9527f}, {0.219923f, 0.00505909f, -0.975239f, 0.0224345f} },
		{ "ES window flicker",       {70.996590f, 6.015063f, -5.423345f}, {-0.712177f, -0.027789f, 0.700885f, -0.027349f} },
		{ "ES fence \"hole\"",       {-18.670401f, 3.432540f, 17.952700f}, {0.138731f, -0.005622f, -0.989478f, -0.040096f} },
		{ "ES strafe problem",       {-4.877779f, 3.371065f, 17.146101f}, {0.994378f, -0.020388f, -0.103789f, -0.002128f} },
		{ "ES catmull showcase",     {-30.011652f, 0.829173f, 27.225056f}, {-0.224099f, 0.012706f, -0.972886f, -0.055162f} },	// enable camera bobbing to see difference
		{ "ES flicker bg. building", {-51.779095f, 3.302949f, 42.258675f}, {-0.922331f, -0.035066f, 0.384432f, -0.014615f} },
	};

	struct MovingObjectDef {
		const char *name;
		const char *filename;
	};
	std::vector<MovingObjectDef> mMovingObjectDefs = {	// name, filename
		{ "Smooth sphere",	"assets/sphere_smooth.obj" },
		{ "Sharp sphere",	"assets/sphere.obj" },
		{ "Soccer ball",	"assets/Soccer_Ball_lores.obj" },
	};


public: // v== cgb::cg_element overrides which will be invoked by the framework ==v
	static const uint32_t cConcurrentFrames = 3u;

	std::string mSceneFileName = "assets/sponza_with_plants_and_terrain.fscene";
	bool mDisableMip = false;
	bool mUseAlphaBlending = false;

	bool mStartCapture;
	int mCaptureNumFrames = 1;
	int mCaptureFramesLeft = 0;

	bool mFlipTexturesInLoader	= false;
	bool mFlipUvWithAssimp		= false;
	bool mFlipManually			= true;

	bool mHideWindowOnLoad = false;

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
			rdoc::labelBuffer(mMatricesUserInputBuffer[i]->handle(), "mMatricesUserInputBuffer", i);
			mMatricesUserInputBuffer[i]->fill(&mMatricesAndUserInput, 0, avk::sync::not_required());
		}
	}

	void update_matrices_and_user_input()
	{
		// Update the matrices in render() because here we can be sure that mQuakeCam's updates of the current frame are available:
		//mMatricesAndUserInput = { mQuakeCam.view_matrix(), mQuakeCam.projection_matrix(), glm::translate(mQuakeCam.translation()), glm::vec4{ 0.f, mNormalMappingStrength, (float)mLightingMode, mAlphaThreshold } };

		static matrices_and_user_input prevFrameMatrices;
		static bool prevFrameValid = false;

		matrices_and_user_input mMatricesAndUserInput = {};
		mMatricesAndUserInput.mViewMatrix	= mQuakeCam.view_matrix();
		mMatricesAndUserInput.mProjMatrix	= mQuakeCam.projection_matrix();
		mMatricesAndUserInput.mCamPos		= glm::translate(mQuakeCam.translation());
		mMatricesAndUserInput.mUserInput	= glm::vec4{ 0.f, mNormalMappingStrength, (float)mLightingMode, mAlphaThreshold };
		mMatricesAndUserInput.mLodBias		= (mLoadBiasTaaOnly && !mAntiAliasing.taa_enabled()) ? 0.f : mLodBias;

		// moving object info
		mMatricesAndUserInput.mActiveMovingObjectMaterialIdx	= mMovingObject.enabled ? mMovingObjectFirstMatIdx + mMovingObject.moverId : -1;
		mMatricesAndUserInput.mMovingObjectModelMatrix			= glm::rotate( glm::translate(glm::mat4(1), mMovingObject.translation), mMovingObject.rotationAngle, glm::vec3(mMovingObject.rotAxisAngle));

		// previous frame info
		if (!prevFrameValid) prevFrameMatrices = mMatricesAndUserInput;	// only copy the partially filled struct for the very first frame

		mMatricesAndUserInput.mPrevFrameProjViewMatrix					= prevFrameMatrices.mProjMatrix * prevFrameMatrices.mViewMatrix;
		mMatricesAndUserInput.mPrevFrameMovingObjectModelMatrix			= prevFrameMatrices.mMovingObjectModelMatrix;
		mMatricesAndUserInput.mPrevFrameActiveMovingObjectMaterialIdx	= prevFrameMatrices.mActiveMovingObjectMaterialIdx;
		mMatricesAndUserInput.mJitterCurrentPrev						= glm::vec4(mCurrentJitter, prevFrameMatrices.mJitterCurrentPrev.x, prevFrameMatrices.mJitterCurrentPrev.y);

		const auto inFlightIndex = gvk::context().main_window()->in_flight_index_for_frame();
		mMatricesUserInputBuffer[inFlightIndex]->fill(&mMatricesAndUserInput, 0, avk::sync::not_required());
		// The cgb::sync::not_required() means that there will be no command buffer which the lifetime has to be handled of.
		// However, we have to ensure to properly sync memory dependency. In this application, this is ensured by the renderpass
		// dependency that is established between VK_SUBPASS_EXTERNAL and subpass 0.

		// store current matrices for use in the next frame
		prevFrameMatrices = mMatricesAndUserInput;
		prevFrameValid = true;
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
			rdoc::labelBuffer(mLightsourcesBuffer[i]->handle(), "mLightsourcesBuffer", i);
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

		auto fif = wnd->number_of_frames_in_flight();
		for (decltype(fif) i=0; i < fif; ++i) {
			auto colorAttachment    = context().create_image(wndRes.x, wndRes.y, IMAGE_FORMAT_COLOR,    1, memory_usage::device, image_usage::general_color_attachment | image_usage::input_attachment);
			colorAttachment->set_target_layout(vk::ImageLayout::eShaderReadOnlyOptimal); // <-- because afterwards, we are going to read from it when applying the post processing effects
			auto depthAttachment    = context().create_image(wndRes.x, wndRes.y, IMAGE_FORMAT_DEPTH,    1, memory_usage::device, image_usage::general_depth_stencil_attachment | image_usage::input_attachment);
			depthAttachment->set_target_layout(vk::ImageLayout::eShaderReadOnlyOptimal); // <-- because afterwards, we are going to read from it when applying the post processing effects
			auto matIdAttachment    = context().create_image(wndRes.x, wndRes.y, IMAGE_FORMAT_MATERIAL, 1, memory_usage::device, image_usage::general_color_attachment | image_usage::input_attachment);
			matIdAttachment->set_target_layout(vk::ImageLayout::eShaderReadOnlyOptimal); // <-- because afterwards, we are going to read from it when applying the post processing effects
			auto velocityAttachment = context().create_image(wndRes.x, wndRes.y, IMAGE_FORMAT_VELOCITY, 1, memory_usage::device, image_usage::general_color_attachment | image_usage::input_attachment);
			velocityAttachment->set_target_layout(vk::ImageLayout::eShaderReadOnlyOptimal); // <-- because afterwards, we are going to read from it when applying the post processing effects

			// label them for Renderdoc
			rdoc::labelImage(colorAttachment->handle(),    "colorAttachment",    i);
			rdoc::labelImage(depthAttachment->handle(),    "depthAttachment",    i);
			rdoc::labelImage(matIdAttachment->handle(),    "matIdAttachment",    i);
			rdoc::labelImage(velocityAttachment->handle(), "velocityAttachment", i);

#if (!FORWARD_RENDERING)
			auto uvNrmAttachment = context().create_image(wndRes.x, wndRes.y, IMAGE_FORMAT_NORMAL,   1, memory_usage::device, image_usage::general_color_attachment | image_usage::input_attachment);
			uvNrmAttachment->set_target_layout(vk::ImageLayout::eShaderReadOnlyOptimal); // <-- because afterwards, we are going to read from it whene applying the post processing effects

			rdoc::labelImage(uvNrmAttachment->handle(), "uvNrmAttachment", i);
#endif

			// Before we are attaching the images to a framebuffer, we have to "wrap" them with an image view:
			auto colorAttachmentView = context().create_image_view(std::move(colorAttachment));
			colorAttachmentView.enable_shared_ownership(); // We are using this attachment in both, the mFramebuffer and the mSkyboxFramebuffer
			auto depthAttachmentView = context().create_depth_image_view(std::move(depthAttachment));
			auto matIdAttachmentView = context().create_image_view(std::move(matIdAttachment));
			auto velocityAttachmentView = context().create_image_view(std::move(velocityAttachment));
#if (!FORWARD_RENDERING)
			auto uvNrmAttachmentView = context().create_image_view(std::move(uvNrmAttachment));
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
						attachment::declare_for(colorAttachmentView,	on_load::load,         color(0) -> color(0),			on_store::store),
						attachment::declare_for(depthAttachmentView,	on_load::clear, depth_stencil() -> depth_stencil(),		on_store::store),
						attachment::declare_for(matIdAttachmentView,	on_load::clear,        color(1) -> color(1),			on_store::store),
						attachment::declare_for(velocityAttachmentView,	on_load::clear,        color(2) -> color(2),			on_store::store),
#else
						attachment::declare_for(colorAttachmentView,	on_load::load,         unused() -> color(0),			on_store::store),
						attachment::declare_for(depthAttachmentView,	on_load::clear, depth_stencil() -> input(0),			on_store::store),
						attachment::declare_for(uvNrmAttachmentView,	on_load::clear,        color(0) -> input(1),			on_store::store),
						attachment::declare_for(matIdAttachmentView,	on_load::clear,        color(1) -> input(2),			on_store::store),
						attachment::declare_for(velocityAttachmentView,	on_load::clear,        color(2) -> unused(),			on_store::store),
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
				std::move(depthAttachmentView),
#if (!FORWARD_RENDERING)
				std::move(uvNrmAttachmentView),
#endif
				std::move(matIdAttachmentView),
				std::move(velocityAttachmentView)
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
			rdoc::beginSection(mSkyboxCommandBuffer[i]->handle(), "Skybox", i);
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
			rdoc::endSection(mSkyboxCommandBuffer[i]->handle());
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
		bool res = false;

		// In Emerald Square v4, all materials with transparent parts are named "*.DoubleSided"
		res |= (std::string::npos != mat.mName.find(".DoubleSided"));

		// In Sponza with plants: materials "leaf" and "Material__57" (vase plants)
		res |= mat.mName == "leaf";
		res |= mat.mName == "Material__57";

		return res;
	}

	bool is_material_twosided(const gvk::material_config &mat) {
		return mat.mTwosided || (std::string::npos != mat.mName.find(".DoubleSided"));	// Emerald-Square leaves are not marked twosided, but can be found by name
	}

	void load_and_prepare_scene() // up to the point where all draw call data and material data has been assembled
	{
		double t0 = glfwGetTime();
		std::cout << "Loading scene..." << std::endl;

		// Load a scene (in ORCA format) from file:
		auto scene = gvk::orca_scene_t::load_from_file(mSceneFileName, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace | (mFlipUvWithAssimp ? aiProcess_FlipUVs : 0) );

		double tLoad = glfwGetTime();

		// print scene graph
		//print_scene_debug_info(scene);
		//print_material_debug_info(scene);

		// Change the materials of "terrain" and "debris", enable tessellation for them, and set displacement scaling:
		helpers::set_terrain_material_config(scene);

		// set custom data to indicate two-sided materials	// TODO: move this into helpers::
		for (auto& model : scene->models()) {
			auto meshIndices = model.mLoadedModel->select_all_meshes();
			for (auto i : meshIndices) {
				auto m = model.mLoadedModel->material_config_for_mesh(i);
				m.mCustomData[3] = is_material_twosided(m) ? 1.0f : 0.0f;
				model.mLoadedModel->set_material_config_for_mesh(i, m);
			}
		}

		
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

		// for cache efficiency, we want to render meshgroups using the same material in sequence, so: walk the materials, find matching meshes, build meshgroup
		for (const auto& pair : distinctMaterialsOrca) {
			const int materialIndex = static_cast<int>(distinctMaterialConfigs.size());
			distinctMaterialConfigs.push_back(pair.first);
			assert (static_cast<size_t>(materialIndex + 1) == distinctMaterialConfigs.size());

			bool materialHasTransparency = has_material_transparency(pair.first);

			// walk the meshdefs having the current material (over ALL orca-models)
			for (const auto& modelAndMeshIndices : pair.second) {
				// Gather the model reference and the mesh indices of the same material in a vector:
				auto& modelData = scene->model_at_index(modelAndMeshIndices.mModelIndex);
				std::vector<std::tuple<std::reference_wrapper<const gvk::model_t>, std::vector<size_t>>> modelRefAndMeshIndices = { std::make_tuple(std::cref(modelData.mLoadedModel), modelAndMeshIndices.mMeshIndices) };
				helpers::exclude_a_curtain(modelRefAndMeshIndices);
				if (modelRefAndMeshIndices.empty()) continue;

				// walk the individual meshes (actually these correspond to "mesh groups", referring to the same meshId) in the same-material-group
				for (auto& modelRefMeshIndicesPair : modelRefAndMeshIndices) {
					for (auto meshIndex : std::get<std::vector<size_t>>(modelRefMeshIndicesPair)) {
						counter++;
						std::cout << "Parsing scene " << counter << "\r"; std::cout.flush();

						std::vector<size_t> tmpMeshIndexVector = { meshIndex };
						std::vector<std::tuple<std::reference_wrapper<const gvk::model_t>, std::vector<size_t>>> selection = { std::make_tuple(std::cref(modelData.mLoadedModel), tmpMeshIndexVector) };


						// build a meshgroup, covering ALL orca-instances
						// TODO: this way of buffer-building is not very efficient; better build buffers for ALL meshes in one go...

						// get the data of the current orca-mesh(group)
						auto [positions, indices] = gvk::get_vertices_and_indices(selection);
						auto texCoords            = mFlipManually ? gvk::get_2d_texture_coordinates_flipped(selection, 0) : gvk::get_2d_texture_coordinates(selection, 0);
						auto normals              = gvk::get_normals   (selection);
						auto tangents             = gvk::get_tangents  (selection);
						auto bitangents           = gvk::get_bitangents(selection);

						Meshgroup mg;
						mg.numIndices      = static_cast<uint32_t>(indices.size());
						mg.numVertices     = static_cast<uint32_t>(positions.size());
						mg.baseIndex       = static_cast<uint32_t>(mSceneData.mIndices.size());
						mg.baseVertex      = static_cast<uint32_t>(mSceneData.mPositions.size());
						mg.materialIndex   = materialIndex;
						mg.hasTransparency = materialHasTransparency;
						// for debugging:
						mg.orcaModelId = static_cast<uint32_t>(modelAndMeshIndices.mModelIndex);
						mg.orcaMeshId  = static_cast<uint32_t>(meshIndex);

						// append the data of the current mesh(group) to the scene vectors
						gvk::append_indices_and_vertex_data(
							gvk::additional_index_data (mSceneData.mIndices,	[&]() { return indices;	 }),
							gvk::additional_vertex_data(mSceneData.mPositions,	[&]() { return positions; })
						);
						gvk::insert_into(mSceneData.mTexCoords,  texCoords);
						gvk::insert_into(mSceneData.mNormals,    normals);
						gvk::insert_into(mSceneData.mTangents,   tangents);
						gvk::insert_into(mSceneData.mBitangents, bitangents);

						if (mg.hasTransparency) mSceneData.mNumTransparentMeshgroups++; else mSceneData.mNumOpaqueMeshgroups++;

						// collect all the instances of the meshgroup
						auto in_instance_transforms = get_mesh_instance_transforms(modelData.mLoadedModel, static_cast<int>(meshIndex), glm::mat4(1));
						// for each orca-instance of the loaded model, apply the instance transform and store the final transforms in the meshgroup
						for (size_t i = 0; i < modelData.mInstances.size(); ++i) {
							auto baseTransform = gvk::matrix_from_transforms(modelData.mInstances[i].mTranslation, glm::quat(modelData.mInstances[i].mRotation), modelData.mInstances[i].mScaling);
							for (auto t = 0; t < in_instance_transforms.size(); ++t) {
								MeshgroupPerInstanceData pid;
								pid.modelMatrix = baseTransform * in_instance_transforms[t];
								mg.perInstanceData.push_back(pid);
							}
						}

						mSceneData.mMeshgroups.push_back(mg);
					}
				}
			}
		}
		std::cout << std::endl;

		// sort meshgroups by transparency (we want to render opaque objects first)
		std::sort(mSceneData.mMeshgroups.begin(), mSceneData.mMeshgroups.end(), [](Meshgroup a, Meshgroup b) { return static_cast<int>(a.hasTransparency) < static_cast<int>(b.hasTransparency); });

		// create all the buffers - for details, see upload_materials_and_vertex_data_to_gpu()
		size_t numMeshgroups = mSceneData.mMeshgroups.size();
		size_t numInstances = 0;
		for (auto &mg : mSceneData.mMeshgroups) numInstances += mg.perInstanceData.size();
		mSceneData.mIndexBuffer           = gvk::context().create_buffer(avk::memory_usage::device, {}, avk::index_buffer_meta::create_from_data(mSceneData.mIndices));
		mSceneData.mPositionsBuffer       = gvk::context().create_buffer(avk::memory_usage::device, {}, avk::vertex_buffer_meta::create_from_data(mSceneData.mPositions).describe_only_member(mSceneData.mPositions[0], avk::content_description::position));
		mSceneData.mTexCoordsBuffer       = gvk::context().create_buffer(avk::memory_usage::device, {}, avk::vertex_buffer_meta::create_from_data(mSceneData.mTexCoords));
		mSceneData.mNormalsBuffer         = gvk::context().create_buffer(avk::memory_usage::device, {}, avk::vertex_buffer_meta::create_from_data(mSceneData.mNormals));
		mSceneData.mTangentsBuffer        = gvk::context().create_buffer(avk::memory_usage::device, {}, avk::vertex_buffer_meta::create_from_data(mSceneData.mTangents));
		mSceneData.mBitangentsBuffer      = gvk::context().create_buffer(avk::memory_usage::device, {}, avk::vertex_buffer_meta::create_from_data(mSceneData.mBitangents));
		mSceneData.mMaterialIndexBuffer   = gvk::context().create_buffer(avk::memory_usage::device, {}, avk::storage_buffer_meta::create_from_size(numMeshgroups * sizeof(uint32_t)));
		mSceneData.mAttribBaseIndexBuffer = gvk::context().create_buffer(avk::memory_usage::device, {}, avk::storage_buffer_meta::create_from_size(numMeshgroups * sizeof(uint32_t)));
		mSceneData.mAttributesBuffer      = gvk::context().create_buffer(avk::memory_usage::device, {}, avk::storage_buffer_meta::create_from_size(numInstances  * sizeof(MeshgroupPerInstanceData)));
		mSceneData.mDrawCommandsBuffer    = gvk::context().create_buffer(avk::memory_usage::device, {}, avk::indirect_buffer_meta::create_from_num_elements_for_draw_indexed_indirect(numMeshgroups));
		
		rdoc::labelBuffer(mSceneData.mIndexBuffer          ->handle(), "scene_IndexBuffer");
		rdoc::labelBuffer(mSceneData.mPositionsBuffer      ->handle(), "scene_PositionsBuffer");
		rdoc::labelBuffer(mSceneData.mTexCoordsBuffer      ->handle(), "scene_TexCoordsBuffer");
		rdoc::labelBuffer(mSceneData.mNormalsBuffer        ->handle(), "scene_NormalsBuffer");
		rdoc::labelBuffer(mSceneData.mTangentsBuffer       ->handle(), "scene_TangentsBuffer");
		rdoc::labelBuffer(mSceneData.mBitangentsBuffer     ->handle(), "scene_BitangentsBuffer");
		rdoc::labelBuffer(mSceneData.mMaterialIndexBuffer  ->handle(), "scene_MaterialIndexBuffer");
		rdoc::labelBuffer(mSceneData.mAttribBaseIndexBuffer->handle(), "scene_AttribBaseIndexBuffer");
		rdoc::labelBuffer(mSceneData.mAttributesBuffer     ->handle(), "scene_AttributesBuffer");
		rdoc::labelBuffer(mSceneData.mDrawCommandsBuffer   ->handle(), "scene_DrawCommandsBuffer");

		double tParse = glfwGetTime();

		// load and add moving objects
		std::cout << "Loading extra models" << std::endl;
		mMovingObjectFirstMatIdx = static_cast<int>(distinctMaterialConfigs.size());
		for (size_t iMover = 0; iMover < mMovingObjectDefs.size(); iMover++)
		{
			// FIXME - this only works for objects with 1 mesh (at least only the first mesh is rendered)
			auto &objdef = mMovingObjectDefs[iMover];
			auto filename = objdef.filename;
			if (!std::filesystem::exists(filename)) {
				LOG_WARNING("Object file \"" + std::string(filename) + "\" does not exist - falling back to default sphere)");
				filename = "assets/sphere.obj";
			}
			auto model = gvk::model_t::load_from_file(filename, aiProcess_Triangulate | aiProcess_CalcTangentSpace);

			const int materialIndex = static_cast<int>(distinctMaterialConfigs.size());
			auto material = model->material_config_for_mesh(0);
			if (material.mAmbientReflectivity == glm::vec4(0)) material.mAmbientReflectivity = glm::vec4(glm::vec3(0.1f), 0); // give it some ambient reflectivity if it has none
			material.mCustomData[0] = iMover + 1.f; // moving object id + 1
			distinctMaterialConfigs.push_back(material);

			auto selection = make_models_and_meshes_selection(model, 0);
			auto [vertices, indices] = gvk::get_vertices_and_indices(selection);
			auto texCoords = mFlipManually ? gvk::get_2d_texture_coordinates_flipped(selection, 0)
										   : gvk::get_2d_texture_coordinates        (selection, 0);
			//std::vector<glm::vec2> texCoords(vertices.size(), glm::vec2(0, 0)); // was necessary instead of above line due to a bug in gvk::model.hpp

			auto normals = gvk::get_normals(selection);
			auto tangents = gvk::get_tangents(selection);
			auto bitangents = gvk::get_bitangents(selection);

			std::vector<push_constant_data_per_drawcall> pcvec = { push_constant_data_per_drawcall{ glm::mat4(1), materialIndex } };
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
			ref.hasTransparency = false;
		}
		// --end moving objects


		// Convert the material configs (that were gathered above) into a GPU-compatible format:
		// "GPU-compatible format" in this sense means that we'll get two things out of the call to `convert_for_gpu_usage`:
		//   1) Material data in a properly aligned format, suitable for being uploaded into a GPU buffer (but not uploaded into a buffer yet!)
		//   2) Image samplers (which contain images and samplers) of all the used textures, already uploaded to the GPU.
		std::tie(mMaterialData, mImageSamplers) = gvk::convert_for_gpu_usage(
			distinctMaterialConfigs, false,
			mFlipTexturesInLoader,
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
		rdoc::labelBuffer(mMaterialBuffer->handle(), "mMaterialBuffer");

		// ac: get the dir light source from the scene file
		auto dirLights = scene->directional_lights();
		if (dirLights.size()) {
			mDirLight.dir       = dirLights[0].mDirection;
			mDirLight.intensity = dirLights[0].mIntensity;
			mDirLight.boost     = 1.f;
		} else {
			// if nothing is in the scene file, use the default from helpers::
			auto lights = helpers::get_lights();
			int idxDir  = helpers::get_lightsource_type_begin_index(gvk::lightsource_type::directional);
			int idxDir2 = helpers::get_lightsource_type_end_index(gvk::lightsource_type::directional);
			int idxAmb  = helpers::get_lightsource_type_begin_index(gvk::lightsource_type::ambient);
			int idxAmb2 = helpers::get_lightsource_type_end_index(gvk::lightsource_type::ambient);
			if (idxDir < idxDir2) {
				mDirLight.dir = lights[idxDir].mDirection;
				mDirLight.intensity = lights[idxDir].mColor;
				mDirLight.boost     = 1.f;
			}
			if (idxAmb < idxAmb2) {
				//mAmbLight.col		= lights[idxAmb].mColor;
				//mAmbLight.boost     = 1.f;
			}
		}


		double tFini  = glfwGetTime();
		printf("Loading took %.1f sec, parsing %.1f sec, rest  %.1f sec => total = %.1f sec\n", tLoad-t0, tParse-tLoad, tFini-tParse, tFini-t0);
	}

	void upload_materials_and_vertex_data_to_gpu()
	{
		std::cout << "Uploading data to GPU..."; std::cout.flush();

		// All of the following are submitted to the same queue (due to cgb::device_queue_selection_strategy::prefer_everything_on_single_queue)
		// That also means that this is the same queue which is used for graphics rendering.
		// Furthermore, this means that it is sufficient to establish a memory barrier after the last call to cgb::fill. 

		mSceneData.mIndexBuffer     ->fill(mSceneData.mIndices.data(),    0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
		mSceneData.mPositionsBuffer ->fill(mSceneData.mPositions.data(),  0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
		mSceneData.mTexCoordsBuffer ->fill(mSceneData.mTexCoords.data(),  0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
		mSceneData.mNormalsBuffer   ->fill(mSceneData.mNormals.data(),    0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
		mSceneData.mTangentsBuffer  ->fill(mSceneData.mTangents.data(),   0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
		mSceneData.mBitangentsBuffer->fill(mSceneData.mBitangents.data(), 0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));

		// build materialIndexBuffer (materialIndexBuffer[i] holds the material index for meshgroup i), attributes / attrib base index buffers, and draw commands buffer
		std::vector<uint32_t> matIdxData    (mSceneData.mMeshgroups.size());
		std::vector<uint32_t> attribBaseData(mSceneData.mMeshgroups.size());
		std::vector<MeshgroupPerInstanceData> attributesData;
		std::vector<VkDrawIndexedIndirectCommand> drawcommandsData(mSceneData.mMeshgroups.size());
		for (auto i = 0; i < mSceneData.mMeshgroups.size(); ++i) {
			auto &mg = mSceneData.mMeshgroups[i];
			matIdxData[i] = mg.materialIndex;
			attribBaseData[i] = static_cast<uint32_t>(attributesData.size());
			gvk::insert_into(attributesData, mg.perInstanceData);
			VkDrawIndexedIndirectCommand dc;
			dc.indexCount    = mg.numIndices;
			dc.instanceCount = static_cast<uint32_t>(mg.perInstanceData.size());
			dc.firstIndex    = mg.baseIndex;
			dc.vertexOffset  = 0;	// already taken care of
			dc.firstInstance = 0;
			drawcommandsData[i] = dc;
		}
		// and upload
		mSceneData.mMaterialIndexBuffer  ->fill(matIdxData.data(),        0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
		mSceneData.mAttribBaseIndexBuffer->fill(attribBaseData.data(),    0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
		mSceneData.mAttributesBuffer     ->fill(attributesData.data(),    0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
		mSceneData.mDrawCommandsBuffer   ->fill(drawcommandsData.data(),  0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));

		// upload data for other models
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

		std::cout << " done" << std::endl;
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
			push_constant_binding_data { shader_type::all, 0, sizeof(push_constant_data_for_dii) }, // We also have to declare that we're going to submit push constants
			descriptor_binding(0, 0, mMaterialBuffer),	// As far as used resources are concerned, we need the materials buffer (type: vk::DescriptorType::eStorageBuffer),
			descriptor_binding(0, 1, mImageSamplers),		// multiple images along with their sampler (array of vk::DescriptorType::eCombinedImageSampler),
			descriptor_binding(0, 2, mSceneData.mMaterialIndexBuffer),		// per meshgroup: material index
			descriptor_binding(0, 3, mSceneData.mAttribBaseIndexBuffer),	// per meshgroup: attributes base index
			descriptor_binding(0, 4, mSceneData.mAttributesBuffer),			// per mesh:      attributes (model matrix)
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
			push_constant_binding_data { shader_type::all, 0, sizeof(push_constant_data_for_dii) },
			// Further bindings which contribute to the pipeline layout:
			//   Descriptor sets 0 and 1 are exactly the same as in mPipelineFirstPass
			descriptor_binding(0, 0, mMaterialBuffer),
			descriptor_binding(0, 1, mImageSamplers),
			descriptor_binding(0, 2, mSceneData.mMaterialIndexBuffer),		// per meshgroup: material index
			descriptor_binding(0, 3, mSceneData.mAttribBaseIndexBuffer),	// per meshgroup: attributes base index
			descriptor_binding(0, 4, mSceneData.mAttributesBuffer),			// per mesh:      attributes (model matrix)
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

		const char * vert_shader_name = "shaders/transform_and_pass_on.vert";
		const char * frag_shader_name = "shaders/fwd_geometry.frag";

		mPipelineFwdOpaque = context().create_graphics_pipeline_for(
			// Specify which shaders the pipeline consists of (type is inferred from the extension):
			vert_shader_name,
			fragment_shader(frag_shader_name).set_specialization_constant(SPECCONST_ID_TRANSPARENCY, uint32_t{ SPECCONST_VAL_OPAQUE }), //  opaque pass
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
			push_constant_binding_data { shader_type::all, 0, sizeof(push_constant_data_for_dii) }, // We also have to declare that we're going to submit push constants
			descriptor_binding(0, 0, mMaterialBuffer),	// As far as used resources are concerned, we need the materials buffer (type: vk::DescriptorType::eStorageBuffer),
			descriptor_binding(0, 1, mImageSamplers),		// multiple images along with their sampler (array of vk::DescriptorType::eCombinedImageSampler),
			descriptor_binding(0, 2, mSceneData.mMaterialIndexBuffer),		// per meshgroup: material index
			descriptor_binding(0, 3, mSceneData.mAttribBaseIndexBuffer),	// per meshgroup: attributes base index
			descriptor_binding(0, 4, mSceneData.mAttributesBuffer),			// per mesh:      attributes (model matrix)
			descriptor_binding(1, 0, mMatricesUserInputBuffer[0]),
			descriptor_binding(1, 1, mLightsourcesBuffer[0])
		);

		// TODO-DII: other pipes

		// this is almost the same, except for the specialization constant, alpha blending (and backface culling? depth write?)
		mPipelineFwdTransparent = context().create_graphics_pipeline_for(
			vert_shader_name,
			fragment_shader(frag_shader_name).set_specialization_constant(SPECCONST_ID_TRANSPARENCY, uint32_t{ SPECCONST_VAL_TRANSPARENT }), // transparent pass

			//cfg::color_blending_config::enable_alpha_blending_for_all_attachments(),
			cfg::color_blending_config::enable_alpha_blending_for_attachment(0),
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
			push_constant_binding_data { shader_type::all, 0, sizeof(push_constant_data_for_dii) }, // We also have to declare that we're going to submit push constants
			descriptor_binding(0, 0, mMaterialBuffer),	// As far as used resources are concerned, we need the materials buffer (type: vk::DescriptorType::eStorageBuffer),
			descriptor_binding(0, 1, mImageSamplers),		// multiple images along with their sampler (array of vk::DescriptorType::eCombinedImageSampler),
			descriptor_binding(0, 2, mSceneData.mMaterialIndexBuffer),		// per meshgroup: material index
			descriptor_binding(0, 3, mSceneData.mAttribBaseIndexBuffer),	// per meshgroup: attributes base index
			descriptor_binding(0, 4, mSceneData.mAttributesBuffer),			// per mesh:      attributes (model matrix)
			descriptor_binding(1, 0, mMatricesUserInputBuffer[0]),
			descriptor_binding(1, 1, mLightsourcesBuffer[0])
		);

		// alternative pipeline - render transparency with simple alpha test only, no blending
		mPipelineFwdTransparentNoBlend = context().create_graphics_pipeline_for(
			vert_shader_name,
			fragment_shader(frag_shader_name).set_specialization_constant(SPECCONST_ID_TRANSPARENCY, uint32_t{ SPECCONST_VAL_TRANSPARENT }), // transparent pass

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
			// TODO-DII: do we still need push constants at all? - maybe for movers, later
			push_constant_binding_data { shader_type::all, 0, sizeof(push_constant_data_for_dii) }, // We also have to declare that we're going to submit push constants
			descriptor_binding(0, 0, mMaterialBuffer),	// As far as used resources are concerned, we need the materials buffer (type: vk::DescriptorType::eStorageBuffer),
			descriptor_binding(0, 1, mImageSamplers),		// multiple images along with their sampler (array of vk::DescriptorType::eCombinedImageSampler),
			descriptor_binding(0, 2, mSceneData.mMaterialIndexBuffer),		// per meshgroup: material index
			descriptor_binding(0, 3, mSceneData.mAttribBaseIndexBuffer),	// per meshgroup: attributes base index
			descriptor_binding(0, 4, mSceneData.mAttributesBuffer),			// per mesh:      attributes (model matrix)
			descriptor_binding(1, 0, mMatricesUserInputBuffer[0]),
			descriptor_binding(1, 1, mLightsourcesBuffer[0])
		);

	}

	// stuff that will eventually go into avk::command_buffer_t::draw_indexed_indirect later
	void draw_scene_indexed_indirect(avk::command_buffer &cmd, uint32_t firstDraw, uint32_t numDraws) {
		cmd->draw_indexed_indirect(
			*mSceneData.mDrawCommandsBuffer,
			*mSceneData.mIndexBuffer,
			numDraws,
			vk::DeviceSize{ firstDraw * sizeof(VkDrawIndexedIndirectCommand) },
			static_cast<uint32_t>(sizeof(vk::DrawIndexedIndirectCommand)),	// cast is critical, won't compile without; draw_indexed suffers from the same problem btw.
			*mSceneData.mPositionsBuffer,
			*mSceneData.mTexCoordsBuffer,
			*mSceneData.mNormalsBuffer,
			*mSceneData.mTangentsBuffer,
			*mSceneData.mBitangentsBuffer
		);
	}

#if !RECORD_CMDBUFFER_IN_RENDER
	// Record actual draw calls for all the drawcall-data that we have
	// gathered in load_and_prepare_scene() into a command buffer
	void record_command_buffer_for_models() {
		using namespace avk;
		using namespace gvk;

		auto* wnd = gvk::context().main_window();
		//auto& commandPool = context().get_command_pool_for_reusable_command_buffers(*mQueue);
		auto& commandPool = context().get_command_pool_for_resettable_command_buffers(*mQueue); // ac: we may need to re-record

		auto fif = wnd->number_of_frames_in_flight();
		for (decltype(fif) i=0; i < fif; ++i) {
			if (!(mModelsCommandBuffer[i].has_value())) mModelsCommandBuffer[i] = commandPool->alloc_command_buffer();
			record_single_command_buffer_for_models(mModelsCommandBuffer[i], i);
		}
	}
#endif

	void record_single_command_buffer_for_models(avk::command_buffer &commandBuffer, gvk::window::frame_id_t fif)
	{
#if FORWARD_RENDERING
		const auto &firstPipe  = mPipelineFwdOpaque;
		const auto &secondPipe = mUseAlphaBlending ? mPipelineFwdTransparent : mPipelineFwdTransparentNoBlend;
#else
		const auto &firstPipe  = mPipelineFirstPass;
		const auto &secondPipe = mPipelineLightingPass;
#endif

		using namespace avk;
		using namespace gvk;

		commandBuffer->begin_recording();
		rdoc::beginSection(commandBuffer->handle(), "Render models", fif);
		helpers::record_timing_interval_start(commandBuffer->handle(), fmt::format("mModelsCommandBuffer{} time", fif));

		// Bind the descriptors for descriptor sets 0 and 1 before starting to render with a pipeline
		commandBuffer->bind_descriptors(firstPipe->layout(), mDescriptorCache.get_or_create_descriptor_sets({ // They must match the pipeline's layout (per set!) exactly.
			descriptor_binding(0, 0, mMaterialBuffer),
			descriptor_binding(0, 1, mImageSamplers),
			descriptor_binding(0, 2, mSceneData.mMaterialIndexBuffer),
			descriptor_binding(0, 3, mSceneData.mAttribBaseIndexBuffer),
			descriptor_binding(0, 4, mSceneData.mAttributesBuffer),
			descriptor_binding(1, 0, mMatricesUserInputBuffer[fif]),
			descriptor_binding(1, 1, mLightsourcesBuffer[fif])
		}));

		// Draw using our pipeline for the first pass (Initially this is the only
		//   pass. After task 2 has been implemented, this is the G-Buffer pass):
		commandBuffer->bind_pipeline(firstPipe);
		commandBuffer->begin_render_pass_for_framebuffer(firstPipe->get_renderpass(), mFramebuffer[fif]);

		// draw the opaque parts of the scene (in deferred shading: draw transparent parts too, we don't use blending there anyway)
		push_constant_data_for_dii pushc_dii;
		pushc_dii.mDrawIdOffset = 0;
		commandBuffer->push_constants(firstPipe->layout(), pushc_dii);
#if FORWARD_RENDERING
		draw_scene_indexed_indirect(commandBuffer, 0, mSceneData.mNumOpaqueMeshgroups);
#else
		draw_scene_indexed_indirect(commandBuffer, 0, mSceneData.mNumOpaqueMeshgroups + mSceneData.mNumTransparentMeshgroups);
#endif
		// draw moving object, if any
		if (mMovingObject.enabled) {
			pushc_dii.mDrawIdOffset = -(mMovingObject.moverId + 1);
			commandBuffer->push_constants(firstPipe->layout(), pushc_dii);
			auto &drawCall = mDrawCalls[mMovingObject.moverId];
			commandBuffer->draw_indexed(
				*drawCall.mIndexBuffer,
				*drawCall.mPositionsBuffer,
				*drawCall.mTexCoordsBuffer,
				*drawCall.mNormalsBuffer,
				*drawCall.mTangentsBuffer,
				*drawCall.mBitangentsBuffer
			);
		}

		// Move on to next subpass, synchronizing all data to be written to memory,
		// and to be made visible to the next subpass, which uses it as input.
		commandBuffer->next_subpass();

#if FORWARD_RENDERING
		commandBuffer->bind_pipeline(secondPipe);
		pushc_dii.mDrawIdOffset = mSceneData.mNumOpaqueMeshgroups;
		commandBuffer->push_constants(firstPipe->layout(), pushc_dii);
		draw_scene_indexed_indirect(commandBuffer, mSceneData.mNumOpaqueMeshgroups, mSceneData.mNumTransparentMeshgroups); // FIXME -> wrong gl_DrawId !
#else
		// TODO - is it necessary to rebind descriptors (pipes are compatible) ??
		commandBuffer->bind_pipeline(secondPipe);
		commandBuffer->bind_descriptors(secondPipe->layout(), mDescriptorCache.get_or_create_descriptor_sets({ 
			descriptor_binding(0, 0, mMaterialBuffer),
			descriptor_binding(0, 1, mImageSamplers),
			descriptor_binding(0, 2, mSceneData.mMaterialIndexBuffer),
			descriptor_binding(0, 3, mSceneData.mAttribBaseIndexBuffer),
			descriptor_binding(0, 4, mSceneData.mAttributesBuffer),
			descriptor_binding(1, 0, mMatricesUserInputBuffer[fif]),
			descriptor_binding(1, 1, mLightsourcesBuffer[fif]),
			descriptor_binding(2, 0, mFramebuffer[fif]->image_view_at(1)->as_input_attachment(), shader_type::fragment),
			descriptor_binding(2, 1, mFramebuffer[fif]->image_view_at(2)->as_input_attachment(), shader_type::fragment),
			descriptor_binding(2, 2, mFramebuffer[fif]->image_view_at(3)->as_input_attachment(), shader_type::fragment)
		}));

		const auto& [quadVertices, quadIndices] = helpers::get_quad_vertices_and_indices();
		commandBuffer->draw_indexed(quadIndices, quadVertices);
#endif

		helpers::record_timing_interval_end(commandBuffer->handle(), fmt::format("mModelsCommandBuffer{} time", fif));
		commandBuffer->end_render_pass();
		rdoc::endSection(commandBuffer->handle());
		commandBuffer->end_recording();
	}

	void setup_ui_callback()
	{
		using namespace avk;
		using namespace gvk;
		
		auto imguiManager = current_composition()->element_by_type<imgui_manager>();
		if(nullptr != imguiManager) {
			imguiManager->add_callback([this]() {
				using namespace ImGui;
				using namespace imgui_helper;

				if (!imgui_helper::globalEnable) return;

				static bool firstTimeInit = true;

				static bool showCamPathDefWindow = true;

				static CameraState savedCamState = {};
				if (firstTimeInit) {
					savedCamState = { "saved", mQuakeCam.translation(), mQuakeCam.rotation() };
					mCameraPresets[0].t = savedCamState.t;
					mCameraPresets[0].r = savedCamState.r;
				}


				static auto smplr = context().create_sampler(filter_mode::bilinear, border_handling_mode::clamp_to_edge);
				static auto texIdsAndDescriptions = [&]() {

					// Gather all the framebuffer attachments to display them
					std::vector<std::tuple<std::optional<ImTextureID>, std::string>> v;
					const auto n = mFramebuffer[0]->image_views().size();
					for (size_t i = 0; i < n; ++i) {
						if (mFramebuffer[0]->image_view_at(i)->get_image().config().samples != vk::SampleCountFlagBits::e1) {
							LOG_INFO(fmt::format("Excluding framebuffer attachment #{} from the UI because it has a sample count != 1. Wouldn't be displayed properly, sorry.", i));
							v.emplace_back(std::optional<ImTextureID>{}, fmt::format("Not displaying attachment #{}", i));
						} else {
							if (!is_norm_format(mFramebuffer[0]->image_view_at(i)->get_image().config().format) && !is_float_format(mFramebuffer[0]->image_view_at(i)->get_image().config().format)) {
								LOG_INFO(fmt::format("Excluding framebuffer attachment #{} from the UI because it has format that can not be sampled with a (floating point-type) sampler2D.", i));
								v.emplace_back(std::optional<ImTextureID>{}, fmt::format("Not displaying attachment #{}", i));
							} else {
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

				Begin("Info & Settings");
				SetWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
				SetWindowSize(ImVec2(250.0f, 700.0f), ImGuiCond_FirstUseEver);

				Text("%.3f ms/frame (%.1f FPS)", 1000.0f / GetIO().Framerate, GetIO().Framerate);
				Text("%.3f ms/mSkyboxCommandBuffer", helpers::get_timing_interval_in_ms(fmt::format("mSkyboxCommandBuffer{} time", inFlightIndex)));
				Text("%.3f ms/mModelsCommandBuffer", helpers::get_timing_interval_in_ms(fmt::format("mModelsCommandBuffer{} time", inFlightIndex)));
				Text("%.3f ms/Anti Aliasing", mAntiAliasing.duration());

				// ac: print camera position
				glm::vec3 p = mQuakeCam.translation();
				Text("Camera at %.1f %.1f %.1f", p.x, p.y, p.z);

				static std::vector<float> accum; // accumulate (then average) 10 frames
				accum.push_back(GetIO().Framerate);
				static std::vector<float> values;
				if (accum.size() == 10) {
					values.push_back(std::accumulate(std::begin(accum), std::end(accum), 0.0f) / 10.0f);
					accum.clear();
				}
				if (values.size() > 90) { // Display up to 90(*10) history frames
					values.erase(values.begin());
				}
				PlotLines("FPS", values.data(), static_cast<int>(values.size()), 0, nullptr, 0.0f, FLT_MAX, ImVec2(0.0f, 50.0f));

				TextColored(ImVec4(0.f, .6f, .8f, 1.f), "[F1]: Toggle input-mode");
				TextColored(ImVec4(0.f, .6f, .8f, 1.f), "[LShift+LCtrl]: Slow-motion");
				TextColored(ImVec4(0.f, .6f, .8f, 1.f), "[F2]: Toggle slow-motion");
				TextColored(ImVec4(0.f, .6f, .8f, 1.f), "[F3]: Toggle GUI");

				if (CollapsingHeader("Lights")) {
					SliderInt("max point lights", &mMaxPointLightCount, 0, 98);
					SliderInt("max spot lights", &mMaxSpotLightCount, 0, 11);

					ColorEdit3("dir col", &mDirLight.intensity.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel); SameLine();
					InputFloat3("dir light", &mDirLight.dir.x, "%.2f");
					SliderFloat("dir boost", &mDirLight.boost, 0.f, 1.f);

					ColorEdit3("amb col", &mAmbLight.col.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel); SameLine();
					SliderFloat("amb boost", &mAmbLight.boost, 0.f, 1.f);

					Combo("Lighting", &mLightingMode, "Blinn-Phong\0Color only\0Debug\0Debug 2\0");
				}

				SliderFloat("alpha thres", &mAlphaThreshold, 0.f, 1.f, "%.3f", 2.f);
				if (Checkbox("alpha blending", &mUseAlphaBlending)) mReRecordCommandBuffers = true;
				HelpMarker("When disabled, simple alpha-testing is used.");
				PushItemWidth(60);
				InputFloat("lod bias", &mLodBias, 0.f, 0.f, "%.1f");
				PopItemWidth();
				SameLine();
				Checkbox("taa only##lod bias taa only", &mLoadBiasTaaOnly);

				SliderFloat("Normal Mapping Strength", &mNormalMappingStrength, 0.0f, 1.0f);

				if (CollapsingHeader("Camera")) {
					PushID("CameraStuff");
					Checkbox("move", &mAutoMovement);
					SameLine();
					PushItemWidth(60);
					Combo("##auto movement unit", &mAutoMovementUnits, "/sec\0/frame\0");
					PopItemWidth();
					SameLine();
					if (Button("set&cap")) {
						mAutoMovement = true;
						mStartCapture = true;
					}
					Checkbox("rotation", &mAutoRotate); SameLine(); InputFloat2("##autoRotDeg", &mAutoRotateDegrees.x, "%.1f");
					Checkbox("bobbing", &mAutoBob);
					Separator();
					if (Button("save cam")) { savedCamState.t = mQuakeCam.translation(); savedCamState.r = mQuakeCam.rotation(); };
					SameLine();
					if (Button("restore cam")) { mQuakeCam.set_translation(savedCamState.t); mQuakeCam.set_rotation(savedCamState.r); }

					struct FuncHolder {
						static bool CamPresetGetter(void* data, int idx, const char** out_str) {
							*out_str = reinterpret_cast<wookiee*>(data)->mCameraPresets[idx].name;
							return true;
						}
					};
					static int selectedCamPreset = 0;
					if (Combo("preset", &selectedCamPreset, &FuncHolder::CamPresetGetter, this, static_cast<int>(mCameraPresets.size()))) {
						mQuakeCam.set_translation(mCameraPresets[selectedCamPreset].t);
						mQuakeCam.set_rotation   (mCameraPresets[selectedCamPreset].r);
					}

					if (Button("print cam")) {
						glm::vec3 t = mQuakeCam.translation();
						glm::quat r = mQuakeCam.rotation();
						//printf("{ \"name\", {%gf, %gf, %gf}, {%gf, %gf, %gf, %gf} },\n", t.x, t.y, t.z, r.w, r.x, r.y, r.z);
						printf("{ \"name\", {%ff, %ff, %ff}, {%ff, %ff, %ff, %ff} },\n", t.x, t.y, t.z, r.w, r.x, r.y, r.z);
					}

					Checkbox("follow path", &mCameraSpline.enable);
					if (mCameraSpline.enable && mCameraSpline.spline.camP.size() < 4) mCameraSpline.enable = false; // need min. 4 pts
					if (mCameraSpline.enable && mCameraSpline.tStart == 0.f) mCameraSpline.tStart = static_cast<float>(glfwGetTime());
					SameLine();
					if (Button("reset")) mCameraSpline.tStart = static_cast<float>(glfwGetTime());
					SameLine();
					if (Button("edit")) showCamPathDefWindow = true;

					PopID();
				}
				if (CollapsingHeader("Moving object")) {
					bool old_enabled = mMovingObject.enabled;
					int  old_moverId = mMovingObject.moverId;
					PushID("Movers");
					Checkbox("enable", &mMovingObject.enabled);
					struct FuncHolder {
						static bool MoverGetter(void* data, int idx, const char** out_str) {
							*out_str = reinterpret_cast<wookiee*>(data)->mMovingObjectDefs[idx].name;
							return true;
						}
					};
					Combo("type", &mMovingObject.moverId, &FuncHolder::MoverGetter, this, static_cast<int>(mMovingObjectDefs.size()));
					InputFloat3("start", &mMovingObject.startPos.x); SameLine(); if (Button("cam##start=cam")) { mMovingObject.startPos = mQuakeCam.translation(); }
					InputFloat3("end  ", &mMovingObject.endPos.x  ); SameLine(); if (Button("cam##end=cam"  )) { mMovingObject.endPos   = mQuakeCam.translation(); }
					InputFloat4("rot ax/spd", &mMovingObject.rotAxisAngle.x);
					PushItemWidth(60);
					InputFloat("##speed", &mMovingObject.speed);
					SameLine();
					Combo("speed##unit", &mMovingObject.units, "/sec\0/frame\0");
					PopItemWidth();
					SameLine();
					if (Button("conv")) {
						// FIXME - rotation speed conversion is only ok for constant rotation
						float fps = glm::max(GetIO().Framerate, 0.01f);
						if (mMovingObject.units == 0) {
							mMovingObject.units = 1;
							mMovingObject.speed /= fps;
							mMovingObject.rotAxisAngle.w /= fps;
						} else {
							mMovingObject.units = 0;
							mMovingObject.speed *= fps;
							mMovingObject.rotAxisAngle.w *= fps;
						}
					}
					PushItemWidth(80);
					Combo("repeat", &mMovingObject.repeat, "no\0cycle\0ping-pong\0");
					PopItemWidth();
					SameLine();
					Checkbox("cont rot", &mMovingObject.rotContinous);
					if (Button("reset")) mMovingObject.t = 0.f;
					PopID();

					if (old_enabled != mMovingObject.enabled || old_moverId != mMovingObject.moverId) mReRecordCommandBuffers = true;
				}

				if (rdoc::active()) {
					if (CollapsingHeader("RenderDoc")) {
						Separator();
						if (Button("capture") && !mCaptureFramesLeft && mCaptureNumFrames > 0) mStartCapture = true;
						SameLine();
						PushItemWidth(60);
						InputInt("frames", &mCaptureNumFrames);
						PopItemWidth();
					}
				}

				if (CollapsingHeader("Images")) {
					for (auto& tpl : texIdsAndDescriptions) {
						auto texId = std::get<std::optional<ImTextureID>>(tpl);
						auto description = std::get<std::string>(tpl);
						if (texId.has_value()) {
							Image(texId.value(), ImVec2(192, 108)); SameLine();
						}
						Text(description.c_str());
					}
				}

				End();	// main window



				// camera path def window
				if (showCamPathDefWindow) {
					Begin("Camera path", &showCamPathDefWindow);
					SetWindowPos(ImVec2(500.0f, 10.0f), ImGuiCond_FirstUseEver);		// TODO: fix pos
					SetWindowSize(ImVec2(250.0f, 400.0f), ImGuiCond_FirstUseEver);

					auto &pos = mCameraSpline.spline.camP;
					if (Button("clear")) { pos.clear(); pos.push_back(glm::vec3(0)); }
					PushItemWidth(60);
					InputFloat("max t", &mCameraSpline.spline.cam_t_max, 0.f, 0.f, "%.1f");
					PopItemWidth();
					SameLine(); Checkbox("use arclen", &mCameraSpline.spline.use_arclen);
					SameLine(); if (Button("recalc")) mCameraSpline.spline.calced_arclen = false;
					int delPos = -1;
					int addPos = -1;
					int moveUp = -1;
					int moveDn = -1;
					bool changed = false;
					for (int i = 0; i < static_cast<int>(pos.size()); ++i) {
						PushID(i);
						PushItemWidth(120);
						if (InputFloat3("##pos", &(pos[i].x), "%.2f")) changed = true;
						PopItemWidth();
						SameLine(); if (Button("-")) delPos = i;
						SameLine(); if (Button("+")) addPos = i;
						SameLine(); if (Button("^")) moveUp = i;
						SameLine(); if (Button("v")) moveDn = i;
						SameLine(); if (Button("set")) { pos[i] = mQuakeCam.translation(); changed = true; }
						PopID();
					}
					if (addPos >= 0) { pos.insert(pos.begin() + addPos + 1, glm::vec3(0));	changed = true; }
					if (delPos >= 0) { pos.erase (pos.begin() + delPos);					changed = true; }
					if (moveUp >  0) { std::swap(pos[moveUp - 1], pos[moveUp]);				changed = true; }
					if (moveDn >= 0 && moveDn < static_cast<int>(pos.size())-1) { std::swap(pos[moveDn + 1], pos[moveDn]); changed = true; }
					if (changed) mCameraSpline.spline.calced_arclen = false;

					End();
				}

				firstTimeInit = false;
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

		// hide the window, so we can see the scene loading progress
		GLFWwindow *glfwWin = wnd->handle()->mHandle;
		if (mHideWindowOnLoad) glfwHideWindow(glfwWin);

		// init Renderdoc debug markers
		rdoc::init_debugmarkers(context().device());

		
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

#if !RECORD_CMDBUFFER_IN_RENDER
		record_command_buffer_for_models();
#endif

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
		std::array<image_view*, cConcurrentFrames> srcVelocityImages;
		auto fif = wnd->number_of_frames_in_flight();
		for (decltype(fif) i = 0; i < fif; ++i) {
			srcDepthImages[i]    = &mFramebuffer[i]->image_view_at(1);
//			srcUvNrmImages[i]    = &mFramebuffer[i]->image_view_at(2);
//			srcMatIdImages[i]    = &mFramebuffer[i]->image_view_at(3);
			srcColorImages[i]    = mFramebuffer[i]->image_view_at(0);
			srcVelocityImages[i] = &mFramebuffer[i]->image_view_at(FORWARD_RENDERING ? 3 : 4);
		}

		mAntiAliasing.set_source_image_views(srcColorImages, srcDepthImages, srcVelocityImages);
		current_composition()->add_element(mAntiAliasing);

		if (mHideWindowOnLoad) glfwShowWindow(glfwWin);
	}

	void init_updater_only_once() {
		static bool beenThereDoneThat = false;
		if (beenThereDoneThat) return;
		beenThereDoneThat = true;

		// allow the updater to reload shaders
#if USE_GVK_UPDATER
		mAntiAliasing.init_updater(mUpdater);

		// updating the pipelines crashes when using pre-recorded command buffers! (because pipeline is still in use)
#if RECORD_CMDBUFFER_IN_RENDER

#if FORWARD_RENDERING
		std::vector<avk::graphics_pipeline *> gfx_pipes = { &mPipelineFwdOpaque, &mPipelineFwdTransparent, &mPipelineFwdTransparentNoBlend };
#else
		std::vector<avk::graphics_pipeline *> gfx_pipes = { &mPipelineFirstPass, &mPipelineLightingPass };
#endif

		for (auto ppipe : gfx_pipes) {
			ppipe->enable_shared_ownership();
			mUpdater.on(gvk::shader_files_changed_event(*ppipe)).update(*ppipe);
		}
#endif
		gvk::current_composition()->add_element(mUpdater);
#endif
	}

	void update() override
	{
		init_updater_only_once(); // don't init in initialize, taa hasn't created its pipelines there yet.

		// global GUI toggle
		if (gvk::input().key_pressed(gvk::key_code::f3)) imgui_helper::globalEnable = !imgui_helper::globalEnable;

		// "slow-motion"
		static bool slowMoToggle = false;
		if (gvk::input().key_pressed(gvk::key_code::f2)) slowMoToggle = !slowMoToggle;
		float dt_offset = 0.f;
		if (slowMoToggle || (gvk::input().key_down(gvk::key_code::left_shift) && gvk::input().key_down(gvk::key_code::left_control))) {
			// slow-mo
			Sleep(500);
			dt_offset = 0.5f;
		}

		if (gvk::input().key_pressed(gvk::key_code::b)) {	// bobbing on off shortcut key
			mAutoBob = !mAutoBob;
			if (mAutoBob) mAutoMovement = true;
		}

		const auto inFlightIndex = gvk::context().main_window()->in_flight_index_for_frame();
		if (inFlightIndex == 1 + cConcurrentFrames) {
			mStoredCommandBuffers.clear();
		}

		// auto-movement
		static float tLast = 0.f;
		float t = static_cast<float>(glfwGetTime());
		float dt = tLast ? t - tLast - dt_offset : 0.f;
		tLast = t;

		static float tAccumulated = 0.f;
		tAccumulated += dt; // don't use t directly in animation, problems with "slo-mo" !

		if (mCameraSpline.enable) {
			// FIXME for slow-mo! also need to fix reset
			mQuakeCam.set_translation(mCameraSpline.spline.getPos(t - mCameraSpline.tStart));
		}

		float dtCam = dt;
		static float tCamBob = 0.f;
		if (mAutoMovementUnits == 1) dtCam = 1.f;
		if (!mAutoMovement || !mAutoBob) tCamBob = 0.f;
		if (mAutoMovement && (mAutoRotate || mAutoBob)) {
			glm::quat rotHoriz, rotVert;
			if (mAutoRotate) {
				glm::vec2 rotation = glm::radians(mAutoRotateDegrees) * dtCam;
				rotHoriz = glm::quat_cast(glm::rotate(rotation.x, glm::vec3(0.f, 1.f, 0.f)));
				rotVert = glm::quat_cast(glm::rotate(rotation.y, glm::vec3(1.f, 0.f, 0.f)));
			} else {
				tCamBob += dtCam;
				const float maxBobH = glm::radians(1.0f); // max deviation to one side
				const float maxBobV = glm::radians(.1f);
				const float bobPeriod = 3.f; // for one full "bob" (to-and-fro)
				float rx =  maxBobH * dtCam * cos(fmod(tCamBob * bobPeriod,       2.f * glm::pi<float>()));
				float ry = -maxBobV * dtCam * sin(fmod(tCamBob * bobPeriod * 2.f, 2.f * glm::pi<float>()));
				//printf("rx %f, ry %f, cos(%f)\n", glm::degrees(rx), glm::degrees(ry), glm::degrees(fmod(tCamBob * bobPeriod, 2.f * glm::pi<float>())));
				rotHoriz = glm::quat_cast(glm::rotate(rx, glm::vec3(0.f, 1.f, 0.f)));
				rotVert  = glm::quat_cast(glm::rotate(ry, glm::vec3(1.f, 0.f, 0.f)));
			}
			mQuakeCam.set_rotation(rotHoriz * mQuakeCam.rotation() * rotVert);
		}

		float dtObj = dt;
		if (mMovingObject.units == 1) dtObj = 1.f;
		static float tObjAccumulated = 0.f;
		if (mMovingObject.enabled) {
			tObjAccumulated += dtObj;

			glm::vec3 dir = mMovingObject.endPos - mMovingObject.startPos;
			float len = glm::length(dir);
			if (len > 1e-6) dir /= len;

			mMovingObject.t += dtObj * mMovingObject.speed;
			float effectiveT = mMovingObject.t;
			if (mMovingObject.repeat == 0) {
				if (mMovingObject.t > len) mMovingObject.t = effectiveT = len;	// stay
			} else 	if (mMovingObject.repeat == 1) {
				if (mMovingObject.t > len) mMovingObject.t = effectiveT = 0.f;	// repeat
			} else {
				if (mMovingObject.t > 2.f * len) mMovingObject.t = effectiveT = 0.f;
				else if (mMovingObject.t > len) effectiveT = len - (mMovingObject.t - len);
			}

			mMovingObject.translation = mMovingObject.startPos + effectiveT * dir;
			mMovingObject.rotationAngle = fmod(glm::radians(mMovingObject.rotAxisAngle.w) * (mMovingObject.rotContinous ? tObjAccumulated : effectiveT), 2.f * glm::pi<float>());
		} else {
			tObjAccumulated = 0.f;
		}

		
		// Let Temporal Anti-Aliasing modify the camera's projection matrix:
		auto* mainWnd = gvk::context().main_window();
		auto modifiedProjMat = mAntiAliasing.get_jittered_projection_matrix(mOriginalProjMat, mCurrentJitter, mainWnd->current_frame());
		mAntiAliasing.save_history_proj_matrix(mOriginalProjMat, mainWnd->current_frame());

		mQuakeCam.set_projection_matrix(modifiedProjMat);
		
		if (gvk::input().key_pressed(gvk::key_code::escape) || glfwWindowShouldClose(gvk::context().main_window()->handle()->mHandle)) {
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

#if !RECORD_CMDBUFFER_IN_RENDER
		// re-record command buffers if necessary (this is only in response to gui selections)
		if (mReRecordCommandBuffers) {
			gvk::context().device().waitIdle(); // this is ok in THIS application, not generally though!
			record_command_buffer_for_models();
		}
#endif
		mReRecordCommandBuffers = false;


		// helpers::animate_lights(helpers::get_lights(), gvk::time().absolute_time());
	}

	void render() override
	{
		// renderdoc capturing
		if (mAntiAliasing.trigger_capture() && !mCaptureFramesLeft) mStartCapture = true;
		if (mStartCapture) {
			mStartCapture = false;
			mCaptureFramesLeft = mCaptureNumFrames;
			rdoc::start_capture();
			LOG_INFO("Starting capture for " + std::to_string(mCaptureNumFrames) + " frames");
		} else if (mCaptureFramesLeft) {
			mCaptureFramesLeft--;
			if (!mCaptureFramesLeft) {
				rdoc::end_capture();
				LOG_INFO("Capture finished");
			}
		}


		auto mainWnd = gvk::context().main_window();
		update_matrices_and_user_input();
		update_lightsources();

		auto inFlightIndex = mainWnd->in_flight_index_for_frame();
		mQueue->submit(mSkyboxCommandBuffer[inFlightIndex], std::optional<std::reference_wrapper<avk::semaphore_t>> {});

#if RECORD_CMDBUFFER_IN_RENDER
		auto& commandPool = gvk::context().get_command_pool_for_single_use_command_buffers(*mQueue);
		auto cmdbfr = commandPool->alloc_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
		record_single_command_buffer_for_models(cmdbfr, inFlightIndex);
		mQueue->submit(cmdbfr, std::optional<std::reference_wrapper<avk::semaphore_t>> {});
		mainWnd->handle_lifetime(std::move(cmdbfr));
#else
		mQueue->submit(mModelsCommandBuffer[inFlightIndex], std::optional<std::reference_wrapper<avk::semaphore_t>> {});
#endif

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
#if !RECORD_CMDBUFFER_IN_RENDER
	std::array<avk::command_buffer, cConcurrentFrames> mModelsCommandBuffer;
#endif

	// Different pipelines used for (deferred) shading:
	avk::graphics_pipeline mPipelineFirstPass;
	avk::graphics_pipeline mPipelineLightingPass;

	// Pipelines for forward rendering:
	avk::graphics_pipeline mPipelineFwdOpaque;
	avk::graphics_pipeline mPipelineFwdTransparent;
	avk::graphics_pipeline mPipelineFwdTransparentNoBlend;

	// The elements to handle the post processing effects:
	taa<cConcurrentFrames> mAntiAliasing;

#if USE_GVK_UPDATER
	gvk::updater mUpdater;	// handles shader hot reloading, window resizing
#endif

	int mLightingMode = 0; // 0 = typical; 1 = no lights, just diff color;  2 = debug
	struct { glm::vec3 dir, intensity; float boost; } mDirLight = { {1.f,1.f,1.f}, { 1.f,1.f,1.f }, 1.f }; // this is overwritten with the dir light from the .fscene file
	struct { glm::vec3 col; float boost; } mAmbLight = { {1.f, 1.f, 1.f}, 0.1f };

	float mAlphaThreshold = 0.5f; // 0.001f; // alpha threshold for rendering transparent parts (0.5 ok for alpha-testing, 0.001 for alpha-blending)
	float mLodBias;
	bool mLoadBiasTaaOnly = true;

	glm::vec2 mAutoRotateDegrees = glm::vec2(-45, 0);
	bool mAutoRotate = false;
	bool mAutoBob = false;
	bool mAutoMovement = true;
	int mAutoMovementUnits = 0; // 0 = per sec, 1 = per frame

	int mMovingObjectFirstMatIdx = -1;

	bool mReRecordCommandBuffers = false;

	struct {
		bool      enabled = 0;
		int       moverId = 0;
		glm::vec3 translation = {};
		float     rotationAngle;	// radians
		glm::vec3 startPos = glm::vec3(-5, 1, 0);
		glm::vec3 endPos   = glm::vec3(5,1,0);
		glm::vec4 rotAxisAngle = glm::vec4(0,0,1,-90);		// axis, angle(degr.)
		float     speed    = 5.f;
		float     t = 0.f;
		int       units = 0; // 0 = per sec, 1 = per frame
		int       repeat = 1; // 0 = no, 1 = cycle, 2 = ping-pong
		bool      rotContinous = false;
	} mMovingObject;

	glm::vec2 mCurrentJitter   = {};

	// draw indexed indirect stuff
	struct {
		// vertex attribute buffers
		avk::buffer mIndexBuffer;
		avk::buffer mPositionsBuffer;
		avk::buffer mTexCoordsBuffer;
		avk::buffer mNormalsBuffer;
		avk::buffer mTangentsBuffer;
		avk::buffer mBitangentsBuffer;

		// per-meshgroup buffers
		avk::buffer mMaterialIndexBuffer;
		avk::buffer mAttribBaseIndexBuffer;		// [x] holds the index for mAttributesBuffer, so that mAttributesBuffer[x] is the attributes of the first instance of mesh group x

		// buffers with entries for every mesh-instance
		avk::buffer mAttributesBuffer;

		// buffer holding draw parameters (VkDrawIndexedIndirectCommand)
		avk::buffer mDrawCommandsBuffer;

		// temporary vectors, holding data to be uploaded to the GPU
		std::vector<uint32_t> mIndices;
		std::vector<glm::vec3> mPositions;
		std::vector<glm::vec2> mTexCoords;
		std::vector<glm::vec3> mNormals;
		std::vector<glm::vec3> mTangents;
		std::vector<glm::vec3> mBitangents;

		// the mesh groups
		std::vector<Meshgroup> mMeshgroups;
		uint32_t mNumOpaqueMeshgroups;
		uint32_t mNumTransparentMeshgroups;
	} mSceneData;

	struct {
		bool   enable;
		float  tStart;
		//Spline spline = Spline({glm::vec3(0,0,0),glm::vec3(1,0,0),glm::vec3(2,0,0),glm::vec3(3,0,0)});
		Spline spline = Spline(8.f, { {-1,0,0},{0,0,0},{1,0,0},{1,0,10},{0,0,10},{0,0,0},{0,0,-1} });
	} mCameraSpline;
};

int main(int argc, char **argv) // <== Starting point ==
{
	// Init Renderdoc API
	rdoc::init();

	try {
		// Parse command line
		// first parameter starting without dash is scene filename
		// -- (double-dash) terminates command line, everything after (including scene file name) is ignored
		bool badCmd = false;
		bool disableValidation = false;
		bool forceValidation = false;
		bool disableMip = false;
		bool enableAlphaBlending = false;
		bool disableAlphaBlending = false;
		int  capture_n_frames = 0;
		bool skip_scene_filename = false;
		int window_width  = 1920;
		int window_height = 1080;
		bool hide_window = false;
		std::string sceneFileName = "";
		for (int i = 1; i < argc; i++) {
			if (0 == strcmp("--", argv[i])) {
				break;
			} else if (0 == strncmp("-", argv[i], 1)) {
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
				} else if (0 == _stricmp(argv[i], "-blend")) {
					enableAlphaBlending = true;
					LOG_INFO("Alpha-blending enabled via command line parameter.");
				} else if (0 == _stricmp(argv[i], "-capture")) {
					i++;
					if (i >= argc) { badCmd = true; break; }
					capture_n_frames = atoi(argv[i]);
					if (capture_n_frames < 1) { badCmd = true; break; }
				} else if (0 == _stricmp(argv[i], "-hidewindow")) {
					hide_window = true;
				} else if (0 == _stricmp(argv[i], "-sponza")) {
					skip_scene_filename = true;
				} else if (0 == _stricmp(argv[i], "-test")) {
					skip_scene_filename = true;
					sceneFileName = "../../extras/TestScene/TestScene.fscene";
				} else if (0 == _stricmp(argv[i], "-w")) {
					i++;
					if (i >= argc) { badCmd = true; break; }
					window_width = atoi(argv[i]);
					if (window_width < 1) { badCmd = true; break; }
				} else if (0 == _stricmp(argv[i], "-h")) {
					i++;
					if (i >= argc) { badCmd = true; break; }
					window_height = atoi(argv[i]);
					if (window_height < 1) { badCmd = true; break; }
				} else {
					badCmd = true;
					break;
				}
			} else if (!skip_scene_filename) {
				if (sceneFileName.length()) {
					badCmd = true;
					break;
				}
				sceneFileName = argv[i];
			}
		}
		if (badCmd) {
			printf("Usage: %s [-w <width>] [-h <height>] [-novalidation] [-validation] [-blend] [-noblend] [-nomip] [-hidewindow] [-capture <numFrames>] [-sponza] [-test] [--] [orca scene file path]\n", argv[0]);
			return EXIT_FAILURE;
		}


		// Create a window and open it
		auto mainWnd = gvk::context().create_window("TAA-STAR");
		mainWnd->set_resolution({ window_width, window_height });
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

		// set scene file name and other command line params
		if (sceneFileName.length()) chewbacca.mSceneFileName = sceneFileName;
		chewbacca.mDisableMip = disableMip;
		if (enableAlphaBlending)  chewbacca.mUseAlphaBlending = true;
		if (disableAlphaBlending) chewbacca.mUseAlphaBlending = false;
		chewbacca.mHideWindowOnLoad = hide_window;

		// setup capturing if RenderDoc is active
		if (capture_n_frames > 0 && rdoc::active()) {
			chewbacca.mCaptureNumFrames = capture_n_frames;
			chewbacca.mStartCapture = true;
		}

		auto modifyValidationFunc = [&](gvk::validation_layers &val_layers) {
			// ac: disable or enforce validation layers via command line (renderdoc crashes when they are enabled....)
			val_layers.enable_in_release_mode(forceValidation); // note: in release, this doesn't enable the debug callback, but val.errors are dumped to the console
			if (disableValidation) val_layers.mLayers.clear();
		};

		// request VK_EXT_debug_marker (ONLY if running from RenderDoc!) for object labeling
		gvk::required_device_extensions dev_extensions;
		for (auto ext : rdoc::required_device_extensions()) dev_extensions.add_extension(ext);

		// GO:
		gvk::start(
			gvk::application_name("TAA-STAR"),
			mainWnd,
			chewbacca,
			ui,
			dev_extensions,
			modifyValidationFunc,
			[](vk::PhysicalDeviceFeatures& pdf) {
				pdf.independentBlend  = VK_TRUE;	// request independent blending
				pdf.multiDrawIndirect = VK_TRUE;	// request support for multiple draw indirect
			} 
		);
	}
	catch (gvk::logic_error&) {}
	catch (gvk::runtime_error&) {}
	catch (avk::logic_error&) {}
	catch (avk::runtime_error&) {}

	//if (rdoc::active()) {
	//	printf("Press Enter to close console window\n");
	//	system("pause");
	//}
}
