#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <portable-file-dialogs.h>
#include <string>

#include "rdoc_helper.hpp"
#include "imgui_helper.hpp"
#include "imgui_stdlib.h"
#include "helper_functions.hpp"
#include "taa.hpp"
#include "splines.hpp"
#include "IniUtil.h"
#include "InterpolationCurve.hpp"
#include "BoundingBox.hpp"
#include "ShadowMap.hpp"
#include "FrustumCulling.hpp"
#include "RayTraceCallback.h"

// for implementing/testing new features in gvk/avk, which are not yet merged to master
// if code doesn't compile for you (although you have pulled the latest master on *both* submodules: gears_vk AND auto_vk), set it to 0
#define USE_UNMERGED_FEATURES 0

// use forward rendering? (if 0: use deferred shading)
#define FORWARD_RENDERING 1

// use the gvk updater for shader hot reloading and window resizing ?
#define USE_GVK_UPDATER 1

// re-record (model) command buffer for frame-in-flight always, instead of pre-recording (for all frame-in-flights) only when necessary? (allows shader hot reloading, but is quite a bit slower (~ +2ms for Emerald Square))
#define RERECORD_CMDBUFFERS_ALWAYS 1

// set working directory to path of executable (necessary if taa.vcproj.user is misconfigured or newly created)
#define SET_WORKING_DIRECTORY 1

#define SCENE_DATA_BUFFER_ON_DEVICE 1

// use fences to avoid processing a frame-in-flight that is still executing - note: this shouldn't be necessary! if it is: is there perhaps a GPU resource update in update()
#define ADDITIONAL_FRAME_SYNC_WITH_FENCES 0

// descriptor binding shortcuts
#define SCENE_DRAW_DESCRIPTOR_BINDINGS(fif_)	descriptor_binding(0, 2, mSceneData.mAttributesBuffer),					/* per (global) mesh:	   attributes (model matrix) */		\
												descriptor_binding(0, 3, mSceneData.mDrawnMeshgroupBuffer[fif_]),		/* per (drawn)  meshgroup: material, mesh base index */		\
												descriptor_binding(0, 4, mSceneData.mDrawnMeshAttribIndexBuffer[fif_]),	/* per (drawn)  mesh:      index to attributes buffer*/		\
												descriptor_binding(0, 5, mSceneData.mMeshgroupsLayoutInfoBuffer[fif_]),	/* first transparent meshgroup index                 */


#if ENABLE_SHADOWMAP
#define SHADOWMAP_DESCRIPTOR_BINDINGS(fif_)  descriptor_binding(SHADOWMAP_BINDING_SET, SHADOWMAP_BINDING_SLOT, mShadowmapImageSamplers[fif_]),
#define SHADOWMAP_DESCRIPTOR_BINDINGS_(fif_) descriptor_binding(SHADOWMAP_BINDING_SET, SHADOWMAP_BINDING_SLOT, mShadowmapImageSamplers[fif_])
#else
#define SHADOWMAP_DESCRIPTOR_BINDINGS(fif_)
#define SHADOWMAP_DESCRIPTOR_BINDINGS_(fif_)
#endif

#if ENABLE_RAYTRACING
#define RAYTRACING_DESCRIPTOR_BINDINGS(fif_)		descriptor_binding(0,  0, mMaterialBuffer),													\
													descriptor_binding(0,  1, mImageSamplers),													\
													descriptor_binding(0,  2, avk::as_uniform_texel_buffer_views(mRtIndexBuffersArray)),		\
													descriptor_binding(0,  3, avk::as_uniform_texel_buffer_views(mRtTexCoordBuffersArray)),		\
													descriptor_binding(0,  4, mRtMaterialIndexBuffer),											\
													descriptor_binding(0,  5, mMatricesUserInputBuffer[fif_]),									\
													descriptor_binding(0,  6, avk::as_uniform_texel_buffer_views(mRtNormalsBuffersArray)),		\
													descriptor_binding(0,  7, mRtPixelOffsetBuffer),											\
													descriptor_binding(0,  8, mRtAnimObjNormalsBufferView[fif_]),								\
													descriptor_binding(0,  9, mRtAnimObjNTBOffsetBuffer[fif_]),									\
													descriptor_binding(0, 10, avk::as_uniform_texel_buffer_views(mRtTangentsBuffersArray)),		\
													descriptor_binding(0, 11, avk::as_uniform_texel_buffer_views(mRtBitangentsBuffersArray)),	\
													descriptor_binding(0, 12, mRtAnimObjTangentsBufferView[fif_]),								\
													descriptor_binding(0, 13, mRtAnimObjBitangentsBufferView[fif_]),							\
													descriptor_binding(0, 14, avk::as_uniform_texel_buffer_views(mRtPositionsBuffersArray)),	\
													descriptor_binding(0, 15, mRtAnimObjPositionsBufferView[fif_]),								\
													descriptor_binding(1,  0, mRtImageViews[fif_]->as_storage_image()),							\
													descriptor_binding(2,  0, mSceneData.mTLASs[fif_])
#endif

// convenience macros
#ifdef _DEBUG
#define IS_DEBUG_BUILD 1
#define IF_DEBUG_BUILD(x) x
#define IF_RELEASE_BUILD(x)
#define IF_DEBUG_BUILD_ELSE(x,y) x
#else
#define IS_DEBUG_BUILD 0
#define IF_DEBUG_BUILD(x)
#define IF_RELEASE_BUILD(x) x
#define IF_DEBUG_BUILD_ELSE(x,y) y
#endif

#define PRINT_DEBUGMARK(msg_) printf("Frame %lld, fif %lld %s\n", gvk::context().main_window()->current_frame(), gvk::context().main_window()->in_flight_index_for_frame(), msg_)

class wookiee : public gvk::invokee, public RayTraceCallback
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
		glm::vec4 mJitterCurrentPrev;

		glm::mat4 mMover_additionalModelMatrix;
		glm::mat4 mMover_additionalModelMatrix_prev;

		glm::mat4 mShadowmapProjViewMatrix[4];
		glm::mat4 mDebugCamProjViewMatrix;
		glm::vec4 mShadowMapMaxDepth;	// for up to 4 cascades

		float mLodBias;
		VkBool32 mAlwaysUseLod0;
		VkBool32 mUseShadowMap;
		float mShadowBias;
		int mShadowNumCascades;
		float pad1, pad2, pad3;
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
	//struct push_constant_data_per_drawcall {
	//	glm::mat4 mModelMatrix;
	//	int mMaterialIndex;
	//};

	// Accumulated drawcall data for rendering the different meshes.
	// Includes an index which refers to an entry in the "list" of materials.
	struct dynamic_object_mesh_data
	{
		avk::buffer mIndexBuffer;
		avk::buffer mPositionsBuffer;
		avk::buffer mTexCoordsBuffer;
		avk::buffer mNormalsBuffer;
		avk::buffer mTangentsBuffer;
		avk::buffer mBitangentsBuffer;
		avk::buffer mBoneWeightsBuffer;
		avk::buffer mBoneIndicesBuffer;

		std::vector<uint32_t> mIndices;
		std::vector<glm::vec3> mPositions;
		std::vector<glm::vec2> mTexCoords;
		std::vector<glm::vec3> mNormals;
		std::vector<glm::vec3> mTangents;
		std::vector<glm::vec3> mBitangents;
		std::vector<glm::vec4> mBoneWeights;
		std::vector<glm::uvec4> mBoneIndices;

		int mMaterialIndex;

#if ENABLE_RAYTRACING
		avk::bottom_level_acceleration_structure mBLAS;
#endif
	};

	struct dynamic_object_part {
		int mMeshIndex;
		glm::mat4 mMeshTransform;
	};

	struct dynamic_object
	{
		std::vector<dynamic_object_mesh_data> mMeshData;
		std::vector<dynamic_object_part> mParts;			// different parts can refer to the same mesh (but with different transforms)

		glm::mat4 mBaseTransform;			// independent of loaded object; neutral position, set by application (mainly scale, transform to foot-point, etc.)

		glm::mat4 mMovementMatrix_current;	// set by application (dynamic movement)
		glm::mat4 mMovementMatrix_prev;		// set by application (dynamic movement, previous frame)

		bool mIsAnimated = false;
		int mActiveAnimation = 0;
		std::vector<gvk::animation> mAnimations;
		std::vector<gvk::animation_clip_data> mAnimClips;
		std::vector<glm::mat4> mBoneMatrices;				// [(mesh0,mat0),...(mesh0,matMAX_BONES-1),(mesh1,mat0),...]		// TODO: is this really per mesh, or should it be per "part" ?
		std::vector<glm::mat4> mBoneMatricesPrev;

		float mAnimTime = 0.f;				// set by application

		bool mIsInAccelerationStructure = false;	// is this included in the current acceleration structure?
		glm::mat4 mMovementMatrix_in_AS;
		uint32_t mRtCustomIndexBase;				// geometry indices custom index base value for this dynObj, actual custom indices = base + meshIndex
	};

	// push constants for DrawIndexedIndirect (also used for single dynamic models)
	struct push_constant_data_for_dii {
		glm::mat4 mMover_baseModelMatrix;
		int       mMover_materialIndex;
		int       mMover_meshIndex;

		int       mDrawType; // 0:scene opaque, 1:scene transparent, negative numbers: moving object id
		int       mShadowMapCascadeToBuild;
	};

	struct push_constant_data_for_rt {
		glm::mat4 mCameraTransform;
		glm::mat4 mCameraViewProjMatrix;
		glm::vec4 mLightDir;
		glm::vec4 mDirLightIntensity;
		glm::vec4 mAmbientLightIntensity;
		float mNormalMappingStrength;
		float mMaxRayLength;
		int  mNumSamples;
		int  mAnimObjFirstMeshId;
		int  mAnimObjNumMeshes;
		uint32_t mDoShadows;		// bit 0: general shadows, bit 1: shadows of transp. objs
		VkBool32 mAugmentTAA;
		VkBool32 mAugmentTAADebug;
		VkBool32 mApproximateLod;
		float pad1, pad2, pad3;
	};

	struct CullingBoundingBox {
		glm::vec4 minPos, maxPos; // only .xyz used
	};

	struct CullingUniforms {
		uint32_t  numMeshgroups;
		uint32_t  numInstances;
		uint32_t  numFrusta;
		uint32_t  drawcmdbuf_FirstTransparentIndex;  // index (not offset!) where transparent draw commands shall start in the produced DrawCommandsBuffer
		glm::vec4 frustumPlanes[5*6];	             // frustum planes, 6 per frustum (frustum #0 = main camera, #1 - #5 = shadow cascades)
	} ubo;

	struct BuildSceneBuffersPushConstants {
		uint32_t frustum;	// 0 = main camera, 1 - 5 = shadow cascades
	};

	struct MeshgroupBasicInfoGpu {
		uint32_t materialIndex;
		uint32_t numInstances;
		uint32_t numIndices;
		uint32_t baseIndex;
		VkBool32 transparent;
	};
	struct MeshgroupPerInstanceData {
		glm::mat4 modelMatrix;
	};
	struct DrawnMeshgroupData {
		uint32_t materialIndex;			// material index
		uint32_t meshIndexBase;			// index of first mesh of the group in MeshAttribOffsetBuffer
	};
	struct MeshgroupsLayoutInfo {	// FIXME - isn't there a better way than to use a separate buffer (or uniform etc.) for this? cannot we just *offset* gl_DrawID ?
		uint32_t transparentMeshgroupsOffset;
	};
	struct Meshgroup {
		uint32_t numIndices;
		uint32_t numVertices;
		uint32_t baseIndex;		// indexbuffer-index  corresponding to the first index of this meshid
		uint32_t baseVertex;	// vertexbuffer-index corresponding to the first index of this meshid (?)
		std::vector<MeshgroupPerInstanceData> perInstanceData;
		int      materialIndex;
		bool     hasTransparency;
		bool     isTwoSided; // redundant, also obtainable via material mCustomData[3]

		// these are mainly for debugging:
		uint32_t orcaModelId;
		uint32_t orcaMeshId;

		BoundingBox boundingBox_untransformed;

		struct {
			avk::buffer rtIndexBuffer, rtPositionBuffer;
			std::vector<uint32_t>  rtIndexData;
			std::vector<glm::vec3> rtPositionData;
			std::vector<glm::vec2> rtTexCoordData;
			std::vector<glm::vec3> rtNormalsData;
			std::vector<glm::vec3> rtTangentsData;
			std::vector<glm::vec3> rtBitangentsData;
		} rayTracingTmp;
	};

	struct CameraState { char name[80];  glm::vec3 t; glm::quat r; };	// ugly char[80] for easier ImGui access...

	std::vector<CameraState> mCameraPresets = {			// NOTE: literal quat constructor = {w,x,y,z} 
		{ "Start" },	// t,r filled in from code
		{ "Origin",                  {0.f, 0.f, 0.f}, {1.f, 0.f, 0.f, 0.f} },
		{ "ES street flicker",       {-18.6704f, 3.43254f, 17.9527f}, {0.219923f, 0.00505909f, -0.975239f, 0.0224345f} },
		{ "ES window flicker",       {70.996590f, 6.015063f, -5.423345f}, {-0.712177f, -0.027789f, 0.700885f, -0.027349f} },
		{ "ES fence \"hole\"",       {-18.670401f, 3.432540f, 17.952700f}, {0.138731f, -0.005622f, -0.989478f, -0.040096f} },
		{ "ES strafe problem",       {-4.877779f, 3.371065f, 17.146101f}, {0.994378f, -0.020388f, -0.103789f, -0.002128f} },
		{ "ES strafe problem 2",     {-43.659477f, 2.072995f, 26.158779f}, {0.985356f, 0.009361f, -0.170158f, 0.001617f} },
		{ "ES catmull showcase",     {-30.011652f, 0.829173f, 27.225056f}, {-0.224099f, 0.012706f, -0.972886f, -0.055162f} },	// enable camera bobbing to see difference
		{ "ES flicker bg. building", {-51.779095f, 3.302949f, 42.258675f}, {-0.922331f, -0.035066f, 0.384432f, -0.014615f} },
	};

	struct MovingObjectDef {
		const char *name;
		const char *filename;
		int        animId;
		glm::mat4  modelMatrix;
	};
	std::vector<MovingObjectDef> mMovingObjectDefs = {
		{ "Smooth sphere",		"assets/sphere_smooth.obj",							-1,	glm::mat4(1) },
		{ "Sharp sphere",		"assets/sphere.obj",								-1,	glm::mat4(1) },
		{ "Soccer ball",		"assets/Soccer_Ball_lores.obj",						-1,	glm::mat4(1) },
		{ "Goblin",				"assets/goblin.dae",								 0,	glm::scale(glm::translate(glm::mat4(1), glm::vec3(0,.95f,0)), glm::vec3(.02f)) },
		{ "Dragon",				"assets/optional/dragon/Dragon_2.5_baked.dae",		 0,	glm::scale(glm::mat4(1), glm::vec3(.1f)) },
		{ "Dude",				"assets/optional/dude/dudeTest03.dae",				 0,	glm::scale(glm::rotate(glm::mat4(1), glm::radians(180.f), glm::vec3(0,1,0)), glm::vec3(3.f)) },
		//{ "Strange weights",	"assets/optional/strange_weights.fbx",				 0,	glm::scale(glm::mat4(1), glm::vec3(.01f)) },
	};

	struct ImageDef {
		const char *name;
		const char *filename;
	};
	std::vector<ImageDef> mImageDefs = {
		{ "Forest sunlight",		"assets/images/Forest_Sunlight_Background-954.jpg" },
		{ "Barcelona",				"assets/images/Barcelona_Spain_Background-1421.jpg" },
		{ "Test card",				"assets/images/Chinese_HDTV_test_card.png" },
		{ "Grid 2560x1440",			"assets/images/thingrid_2560x1440.png" },
		{ "Grid 2560x1440 inv",		"assets/images/thingrid_inv_2560x1440.png" },
	};

	enum class SceneType { Unknown, Sponza, EmeraldSquare, BistroExterior };


public: // v== cgb::cg_element overrides which will be invoked by the framework ==v
	static const uint32_t cConcurrentFrames = 3u;
	static const uint32_t cSwapchainImages  = 3u;

	std::string mSceneFileName = "assets/sponza_with_plants_and_terrain.fscene";
	bool mDisableMip = false;
	bool mUseAlphaBlending = false;
	bool mUpsampling;
	float mUpsamplingFactor = 1.f;

	bool mStartCapture;
	int mCaptureNumFrames = 1;
	int mCaptureFramesLeft = 0;
	bool mStartCaptureEarly = false;

	bool mDoRayTraceTest = false;

	bool mFlipTexturesInLoader	= false;
	bool mFlipUvWithAssimp		= false;
	bool mFlipManually			= true;

	bool mHideWindowOnLoad = false;

	wookiee(avk::queue& aQueue)
		: mQueue{ &aQueue }
		, mAntiAliasing{ &aQueue }
	{}

	void ray_trace_callback(avk::command_buffer &cmd) {
#if ENABLE_RAYTRACING
		//std::cout << "callback...";
		do_raytrace_test(cmd, gvk::context().main_window()->in_flight_index_for_frame(), true);
		//std::cout << "... done" << std::endl;
#endif
	}

	virtual int  getNumRayTraceSamples() { return mRtSamplesPerPixel; };
	virtual void setNumRayTraceSamples(int aNumSamples) { mRtSamplesPerPixel = glm::clamp(aNumSamples, 1, RAYTRACING_MAX_SAMPLES_PER_PIXEL); };
	virtual bool getRayTraceAugmentTaaDebug() { return mRtDebugSparse; };
	virtual void setRayTraceAugmentTaaDebug(bool aDebug) { mRtDebugSparse = aDebug; };


	void prepare_matrices_ubo()
	{
		// Prepare a struct containing all the matrices.
		// This will be updated at the beginning of the render() callback
		//  (which is when we have the updated camera position available)
		matrices_and_user_input mMatricesAndUserInput = { glm::mat4(), glm::mat4(), glm::mat4(), glm::vec4() };

		auto* wnd = gvk::context().main_window();
		auto fif = wnd->number_of_frames_in_flight();
		for (decltype(fif) i=0; i < fif; ++i) {
			mMatricesUserInputBuffer[i] = gvk::context().create_buffer(avk::memory_usage::host_coherent, {}, avk::uniform_buffer_meta::create_from_data(mMatricesAndUserInput));
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
		mMatricesAndUserInput.mViewMatrix			= mQuakeCam.view_matrix();
		mMatricesAndUserInput.mProjMatrix			= mQuakeCam.projection_matrix();
		mMatricesAndUserInput.mCamPos				= glm::translate(mQuakeCam.translation());
		mMatricesAndUserInput.mUserInput			= glm::vec4{ 0.f, mNormalMappingStrength, (float)mLightingMode, mAlphaThreshold };
		mMatricesAndUserInput.mLodBias				= (mLoadBiasTaaOnly && !mAntiAliasing.taa_enabled()) ? 0.f : mLodBias;
		mMatricesAndUserInput.mAlwaysUseLod0		= mAlwaysUseLod0;
		mMatricesAndUserInput.mUseShadowMap			= mShadowMap.enable;
		mMatricesAndUserInput.mShadowBias			= mShadowMap.bias;
		mMatricesAndUserInput.mShadowNumCascades	= mShadowMap.numCascades;

		mMatricesAndUserInput.mDebugCamProjViewMatrix = effectiveCam_proj_matrix() * effectiveCam_view_matrix(); // for drawing frustum

		// previous frame info
		if (!prevFrameValid) prevFrameMatrices = mMatricesAndUserInput;	// only copy the partially filled struct for the very first frame

		mMatricesAndUserInput.mPrevFrameProjViewMatrix					= prevFrameMatrices.mProjMatrix * prevFrameMatrices.mViewMatrix;
		mMatricesAndUserInput.mJitterCurrentPrev						= glm::vec4(mCurrentJitter, prevFrameMatrices.mJitterCurrentPrev.x, prevFrameMatrices.mJitterCurrentPrev.y);

		// dynamic models positioning
		auto &dynObj = mDynObjects[mMovingObject.moverId];
		mMatricesAndUserInput.mMover_additionalModelMatrix      = dynObj.mMovementMatrix_current;
		mMatricesAndUserInput.mMover_additionalModelMatrix_prev = dynObj.mMovementMatrix_prev;

		const auto inFlightIndex = gvk::context().main_window()->in_flight_index_for_frame();

#if ENABLE_SHADOWMAP
		// shadowmap matrices
		mShadowMap.shadowMapUtil.calc(mDirLight.dir, effectiveCam_view_matrix(), effectiveCam_proj_matrix());
		for (int cascade = 0; cascade < mShadowMap.numCascades; ++cascade) {
			mMatricesAndUserInput.mShadowmapProjViewMatrix[cascade] = mShadowMap.shadowMapUtil.projection_matrix(cascade) * mShadowMap.shadowMapUtil.view_matrix();
			mMatricesAndUserInput.mShadowMapMaxDepth[cascade] = mShadowMap.shadowMapUtil.max_depth(cascade);
		}
#endif

		mMatricesUserInputBuffer[inFlightIndex]->fill(&mMatricesAndUserInput, 0, avk::sync::not_required());

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

	void prepare_bone_matrices_buffer(size_t numMatrices)
	{
		auto* wnd = gvk::context().main_window();
		auto fif = wnd->number_of_frames_in_flight();
		for (decltype(fif) i=0; i < fif; ++i) {
			mBoneMatricesBuffer[i] = gvk::context().create_buffer(
				avk::memory_usage::host_coherent, {},
				avk::storage_buffer_meta::create_from_size(numMatrices * sizeof(glm::mat4))
			);
			rdoc::labelBuffer(mBoneMatricesBuffer[i]->handle(), "mBoneMatricesBuffer", i);

			mBoneMatricesPrevBuffer[i] = gvk::context().create_buffer(
				avk::memory_usage::host_coherent, {},
				avk::storage_buffer_meta::create_from_size(numMatrices * sizeof(glm::mat4))
			);
			rdoc::labelBuffer(mBoneMatricesPrevBuffer[i]->handle(), "mBoneMatricesPrevBuffer", i);
		}
	}

	void update_bone_matrices()
	{
		if (!mMovingObject.enabled) return;
		auto &dynObj = mDynObjects[mMovingObject.moverId];
		if (!dynObj.mIsAnimated) return;
		auto anim = dynObj.mAnimations[dynObj.mActiveAnimation];

		// keep matrices of previous frame
		dynObj.mBoneMatricesPrev = dynObj.mBoneMatrices;
		
		for (auto &m : dynObj.mBoneMatrices) m = glm::mat4(0);
		anim.animate_into_strided_target_per_mesh(
			dynObj.mAnimClips[dynObj.mActiveAnimation], 
			static_cast<double>(dynObj.mAnimTime), 
			gvk::bone_matrices_space::model_space,
			dynObj.mBoneMatrices.data(),
			size_t{MAX_BONES} * sizeof(glm::mat4) // Stride between two consecutive sets of bonem atrices
		); // fill bone matrix data (-> dc.mBoneMatrices) in object space

		static bool firstTime = true;
		if (firstTime) {
			// first time init	// FIXME_DYNOBJ - this check is not sufficient
			dynObj.mBoneMatricesPrev = dynObj.mBoneMatrices;
			firstTime = false;
		};

		const auto inFlightIndex = gvk::context().main_window()->in_flight_index_for_frame();
		// must use the overloaded fill method and specify size, because the bone matrix vectors are differently sized for different dynObjects !
		mBoneMatricesBuffer    [inFlightIndex]->fill(dynObj.mBoneMatrices.    data(), 0, 0, dynObj.mBoneMatrices.size()     * sizeof(glm::mat4), avk::sync::not_required());
		mBoneMatricesPrevBuffer[inFlightIndex]->fill(dynObj.mBoneMatricesPrev.data(), 0, 0, dynObj.mBoneMatricesPrev.size() * sizeof(glm::mat4), avk::sync::not_required());
	}

	void update_camera_path_draw_buffer() {
		if (!mCameraSpline.draw) return;
		if (mDrawCamPathPositions_valid) return;

		int nSamples = mCameraSpline.drawNpoints;
		int nCpl     = static_cast<int>(mCameraSpline.interpolationCurve.num_control_points());

		mNumCamPathPositions = nSamples + nCpl; // samples + control points
		if (mNumCamPathPositions > mMaxCamPathPositions) {
			mNumCamPathPositions = mMaxCamPathPositions;
			nSamples = mNumCamPathPositions - nCpl;
		}
		if (nSamples < 2) return;

		// sample points from spline
		std::vector<glm::vec4> samples(mNumCamPathPositions);
		for (int i = 0; i < nSamples; ++i) {
			float t = static_cast<float>(i) / static_cast<float>(nSamples - 1);
			if (mCameraSpline.constantSpeed) t = mCameraSpline.interpolationCurve.mapConstantSpeedTime(t);
			samples[i] = glm::vec4(mCameraSpline.interpolationCurve.value_at(t), 0);
		}
		// control points
		auto &cpl = mCameraSpline.interpolationCurve.control_points();
		for (int i = 0; i < nCpl; ++i) samples[nSamples + i] = glm::vec4(cpl[i], static_cast<float>(i+1));

		mDrawCamPathPositionsBuffer->fill(samples.data(), 0, 0,  samples.size() * sizeof(samples[0]), avk::sync::wait_idle(true)); // wait_idle is ok here, this is only in response to GUI input

		mDrawCamPathPositions_valid = true;
		mReRecordCommandBuffers = true;
	}

	void update_culling_ubo() {
		CullingUniforms ubo;
		ubo.numMeshgroups = static_cast<uint32_t>(mSceneData.mMeshgroups.size());
		ubo.numInstances  = static_cast<uint32_t>(mSceneData.mNumTotalInstances);
		ubo.numFrusta     = mShadowMap.enable ? 1 + mShadowMap.numCascades : 1;
		ubo.drawcmdbuf_FirstTransparentIndex = mSceneData.mMaxOpaqueMeshgroups;

		if (!mSceneData.mCullViewFrustum) ubo.numFrusta = 0;

		// calc frustum planes
		for (uint32_t frustum = 0; frustum < ubo.numFrusta; ++frustum) {
			glm::mat4 pvMatrix = (frustum == 0) ? effectiveCam_proj_matrix() * effectiveCam_view_matrix()
				                                : mShadowMap.shadowMapUtil.projection_matrix(frustum - 1) * mShadowMap.shadowMapUtil.view_matrix();
			FrustumCulling fc(pvMatrix);
			for (int i = 0; i < 6; ++i) {
				ubo.frustumPlanes[6 * frustum + i] = fc.Plane(i);
			}
		}

		const auto fif = gvk::context().main_window()->in_flight_index_for_frame();
		mSceneData.mCullingUniformsBuffer[fif]->fill(&ubo, 0, avk::sync::not_required());
	}

	void rebuild_scene_buffers(gvk::window::frame_id_t fif) {
		rebuild_scene_buffers(fif, fif);
	}

	void rebuild_scene_buffers(gvk::window::frame_id_t from_fif, gvk::window::frame_id_t to_fif) {
		uint32_t numOpaqueMeshgroups = 0;
		uint32_t numTransparentMeshgroups = 0;

		FrustumCulling frustumCulling(effectiveCam_proj_matrix() * effectiveCam_view_matrix());

		std::vector<DrawnMeshgroupData> drawnMeshgroupData;
		std::vector<uint32_t> attribIndexData;
		std::vector<VkDrawIndexedIndirectCommand> drawcommandsData;
		drawnMeshgroupData.reserve(mSceneData.mMeshgroups.size());
		drawcommandsData  .reserve(mSceneData.mMeshgroups.size());
		uint32_t meshIndexBase = 0;
		uint32_t globMeshCount = 0;
		for (auto &mg : mSceneData.mMeshgroups) {
			uint32_t globMeshFirstId = globMeshCount;
			globMeshCount += static_cast<uint32_t>(mg.perInstanceData.size());

			DrawnMeshgroupData meshgroup_data = { static_cast<uint32_t>(mg.materialIndex), static_cast<uint32_t>(attribIndexData.size()) };

			size_t numInstances = 0;
			if (mSceneData.mCullViewFrustum) {
				for (auto iLocalInst = 0; iLocalInst < mg.perInstanceData.size(); ++iLocalInst) {
					auto &insDat = mg.perInstanceData[iLocalInst];
					glm::vec4 p[8];
					mg.boundingBox_untransformed.getTransformedPointsV4(insDat.modelMatrix, p);
					BoundingBox bb;
					bb.calcFromPoints(8, p);
					if (!frustumCulling.CanCull(bb)) {
						attribIndexData.push_back(globMeshFirstId + iLocalInst);
						numInstances++;
					}
				}
			} else {
				for (auto iLocalInst = 0; iLocalInst < mg.perInstanceData.size(); ++iLocalInst) {
					attribIndexData.push_back(globMeshFirstId + iLocalInst);
					numInstances++;
				}
			}

			if (numInstances == 0) continue; // with next mesh group

			drawnMeshgroupData.push_back(meshgroup_data);

			VkDrawIndexedIndirectCommand dc;
			dc.indexCount    = mg.numIndices;
			dc.instanceCount = static_cast<uint32_t>(numInstances);
			dc.firstIndex    = mg.baseIndex;
			dc.vertexOffset  = 0;	// already taken care of
			dc.firstInstance = 0;
			drawcommandsData.push_back(dc);
			if (mg.hasTransparency) numTransparentMeshgroups++; else numOpaqueMeshgroups++;
		}

		std::array<uint32_t, 2> drawCounts = { numOpaqueMeshgroups, numTransparentMeshgroups };

		// Set up synchronization for upload
		// Note: it's not necessary to have a separate memory barrier for each fill, one combined barrier at the last fill is sufficient! (and none at all is required if buffers are host-coherent)
#if SCENE_DATA_BUFFER_ON_DEVICE
		auto makeSyncNone = []() { return avk::sync::with_barriers(gvk::context().main_window()->command_buffer_lifetime_handler(), {}, {}); };
		auto makeSyncAll  = []() {
			return avk::sync::with_barriers(gvk::context().main_window()->command_buffer_lifetime_handler(), {},
				[](avk::command_buffer_t& cb, avk::pipeline_stage srcStage, std::optional<avk::write_memory_access> srcAccess) {
					// data buffers are read in vertex shader, draw command buffer is used for indirect draw
					cb.establish_global_memory_barrier_rw(srcStage,  avk::pipeline_stage::draw_indirect | avk::pipeline_stage::vertex_shader,
						srcAccess, avk::memory_access::indirect_command_data_read_access | avk::memory_access::shader_buffers_and_images_read_access);
				});
		};
#else
		auto makeSyncNone = []() { return avk::sync::not_required(); };
		auto makeSyncAll  = []() { return avk::sync::not_required(); };
#endif

		// and upload
		auto     transp_data      = drawcommandsData.data() + numOpaqueMeshgroups; // start of transparent draw commands
		uint32_t transp_bufOffset = mSceneData.mMaxOpaqueMeshgroups * sizeof(VkDrawIndexedIndirectCommand); // destination for first transparent command in draw commands buffer
		for (decltype(from_fif) i = from_fif; i <= to_fif; ++i) {
			mSceneData.mDrawnMeshgroupBuffer         [i]->fill(drawnMeshgroupData.data(), 0, 0,                drawnMeshgroupData.size() * sizeof(DrawnMeshgroupData),           makeSyncNone());
			mSceneData.mDrawnMeshAttribIndexBuffer   [i]->fill(attribIndexData.data(),    0, 0,                attribIndexData.size()    * sizeof(uint32_t),                     makeSyncNone());
			mSceneData.mMeshgroupsLayoutInfoBuffer   [i]->fill(&numOpaqueMeshgroups,      0, 0,                                            sizeof(uint32_t),                     makeSyncNone());
			mSceneData.mDrawCommandsBuffer           [i]->fill(drawcommandsData.data(),   0, 0,                numOpaqueMeshgroups       * sizeof(VkDrawIndexedIndirectCommand), makeSyncNone());	// opaque
			mSceneData.mDrawCommandsBuffer           [i]->fill(transp_data,               0, transp_bufOffset, numTransparentMeshgroups  * sizeof(VkDrawIndexedIndirectCommand), makeSyncNone());	// transparent
			mSceneData.mDrawCountBuffer              [i]->fill(drawCounts.data(),         0, 0,                drawCounts.size()         * sizeof(uint32_t),                     makeSyncAll());
		}

	}

	void prepare_framebuffers_and_post_process_images()
	{
		using namespace avk;
		using namespace gvk;

		auto* wnd = gvk::context().main_window();

		// Before compiling the actual framebuffer, create its image-attachments:
		const auto loRes = mLoResolution;

		auto fif = wnd->number_of_frames_in_flight();
		for (decltype(fif) i=0; i < fif; ++i) {
			auto colorAttachment    = context().create_image(loRes.x, loRes.y, IMAGE_FORMAT_COLOR,    1, memory_usage::device, image_usage::general_color_attachment | image_usage::input_attachment);
			colorAttachment->set_target_layout(vk::ImageLayout::eShaderReadOnlyOptimal); // <-- because afterwards, we are going to read from it when applying the post processing effects
			auto depthAttachment    = context().create_image(loRes.x, loRes.y, IMAGE_FORMAT_DEPTH,    1, memory_usage::device, image_usage::general_depth_stencil_attachment | image_usage::input_attachment);
			depthAttachment->set_target_layout(vk::ImageLayout::eShaderReadOnlyOptimal); // <-- because afterwards, we are going to read from it when applying the post processing effects
			auto matIdAttachment    = context().create_image(loRes.x, loRes.y, IMAGE_FORMAT_MATERIAL, 1, memory_usage::device, image_usage::general_color_attachment | image_usage::input_attachment | image_usage::general_storage_image);
			matIdAttachment->set_target_layout(vk::ImageLayout::eShaderReadOnlyOptimal); // <-- because afterwards, we are going to read from it when applying the post processing effects
			auto velocityAttachment = context().create_image(loRes.x, loRes.y, IMAGE_FORMAT_VELOCITY, 1, memory_usage::device, image_usage::general_color_attachment | image_usage::input_attachment);
			velocityAttachment->set_target_layout(vk::ImageLayout::eShaderReadOnlyOptimal); // <-- because afterwards, we are going to read from it when applying the post processing effects

			// label them for Renderdoc
			rdoc::labelImage(colorAttachment->handle(),    "colorAttachment",    i);
			rdoc::labelImage(depthAttachment->handle(),    "depthAttachment",    i);
			rdoc::labelImage(matIdAttachment->handle(),    "matIdAttachment",    i);
			rdoc::labelImage(velocityAttachment->handle(), "velocityAttachment", i);

#if (!FORWARD_RENDERING)
			auto uvNrmAttachment = context().create_image(loRes.x, loRes.y, IMAGE_FORMAT_NORMAL,   1, memory_usage::device, image_usage::general_color_attachment | image_usage::input_attachment);
			uvNrmAttachment->set_target_layout(vk::ImageLayout::eShaderReadOnlyOptimal); // <-- because afterwards, we are going to read from it whene applying the post processing effects

			rdoc::labelImage(uvNrmAttachment->handle(), "uvNrmAttachment", i);
#endif

			// Before we are attaching the images to a framebuffer, we have to "wrap" them with an image view:
			auto colorAttachmentView = context().create_image_view(std::move(colorAttachment));
			auto depthAttachmentView = context().create_depth_image_view(std::move(depthAttachment));
			auto matIdAttachmentView = context().create_image_view(std::move(matIdAttachment));
			auto velocityAttachmentView = context().create_image_view(std::move(velocityAttachment));
			colorAttachmentView.enable_shared_ownership(); // We are using this attachment in both, the mFramebuffer and the mSkyboxFramebuffer
			depthAttachmentView.enable_shared_ownership(); // set all that are used in taa to shared ownership too
			matIdAttachmentView.enable_shared_ownership();
			velocityAttachmentView.enable_shared_ownership();
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
						attachment::declare_for(colorAttachmentView,    on_load::load,         color(0) ->             color(0),		on_store::store),
						attachment::declare_for(depthAttachmentView,    on_load::clear, depth_stencil() ->             depth_stencil(),	on_store::store),
						attachment::declare_for(matIdAttachmentView,    on_load::clear,        color(1) ->             unused(),		on_store::store),
						attachment::declare_for(velocityAttachmentView, on_load::clear,        color(2) ->             unused(),		on_store::store),
#else
						attachment::declare_for(colorAttachmentView,	on_load::load,         unused() -> color(0) -> color(0),			on_store::store),
						attachment::declare_for(depthAttachmentView,	on_load::clear, depth_stencil() -> input(0) -> depth_stencil(),		on_store::store),
						attachment::declare_for(uvNrmAttachmentView,	on_load::clear,        color(0) -> input(1) -> unused(),			on_store::store),
						attachment::declare_for(matIdAttachmentView,	on_load::clear,        color(1) -> input(2) -> unused(),			on_store::store),
						attachment::declare_for(velocityAttachmentView,	on_load::clear,        color(2) -> unused() -> unused(),			on_store::store),
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
				shared(mRenderpass),
				shared(   colorAttachmentView),
				shared(   depthAttachmentView),
#if (!FORWARD_RENDERING)
				owned(    uvNrmAttachmentView),
#endif
				shared(   matIdAttachmentView),
				shared(   velocityAttachmentView)
			);
			mFramebuffer[i]->initialize_attachments(sync::wait_idle(true));

			// Create the framebuffer for the skybox:
			mSkyboxFramebuffer[i] = context().create_framebuffer(
				{ attachment::declare_for(colorAttachmentView, on_load::dont_care, color(0), on_store::store) }, 
				shared(   colorAttachmentView)
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
			vertex_shader("shaders/sky_gradient.vert.spv"),
			fragment_shader("shaders/sky_gradient.frag.spv"),
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
					cfg::viewport_depth_scissors_config::from_framebuffer(avk::const_referenced(wnd->backbuffer_at_index(0))), // Set to the dimensions of the main window
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
			mSkyboxCommandBuffer[i]->bind_pipeline(const_referenced(mSkyboxPipeline));
			mSkyboxCommandBuffer[i]->begin_render_pass_for_framebuffer( // Start the renderpass defined for the attachments (in fact, the renderpass is created FROM the attachments, see renderpass_t::create)
				mSkyboxPipeline->get_renderpass(),                   // <-- We'll use the pipeline's renderpass, where we have defined load/store operations.
				mSkyboxFramebuffer[i]
			);
			mSkyboxCommandBuffer[i]->bind_descriptors(mSkyboxPipeline->layout(), mDescriptorCache.get_or_create_descriptor_sets({  // Bind the descriptors which describe resources used by shaders.
				descriptor_binding(0, 0, mMatricesUserInputBuffer[i])                                  // In this case, we have one uniform buffer as resource (we have also declared that during mSkyboxPipeline creation).
			}));
			mSkyboxCommandBuffer[i]->draw_indexed(avk::const_referenced(mSphereIndexBuffer), avk::const_referenced(mSphereVertexBuffer)); // Record the draw call
			mSkyboxCommandBuffer[i]->end_render_pass();
			helpers::record_timing_interval_end(mSkyboxCommandBuffer[i]->handle(), fmt::format("mSkyboxCommandBuffer{} time", i));
			rdoc::endSection(mSkyboxCommandBuffer[i]->handle());
			mSkyboxCommandBuffer[i]->end_recording(); // Done recording. We're not going to modify this command buffer anymore.
		}
	}

	void prepare_shadowmap() {
#if ENABLE_SHADOWMAP
		using namespace avk;
		using namespace gvk;

		auto* wnd = gvk::context().main_window();

		auto numFif = wnd->number_of_frames_in_flight();
		//for (decltype(fif) i = 0; i < fif; ++i) mShadowmapImageSamplers[i].clear();

		std::vector<std::array<avk::image_view, SHADOWMAP_MAX_CASCADES>> depthViews(numFif);

		// create framebuffers & (one) renderpass
		for (int cascade = 0; cascade < mShadowMap.numCascades; ++cascade) {
			if (mShadowmapPerCascade[cascade].mShadowmapFramebuffer[0].has_value()) continue; // already alloced framebuffers for this cascade
			for (decltype(numFif) i = 0; i < numFif; ++i) {
				auto depthAttachment = context().create_image(SHADOWMAP_SIZE, SHADOWMAP_SIZE, IMAGE_FORMAT_SHADOWMAP, 1, memory_usage::device, image_usage::general_depth_stencil_attachment | image_usage::sampled);
				depthAttachment->set_target_layout(vk::ImageLayout::eShaderReadOnlyOptimal); // <-- because afterwards, we are going to read from it
				rdoc::labelImage(depthAttachment->handle(), std::string("shadowDepthAttachment_C" + std::to_string(cascade)).c_str(), i);
				depthViews[i][cascade] = context().create_depth_image_view(std::move(depthAttachment));
				

				if (!mShadowmapRenderpass.has_value()) {
					mShadowmapRenderpass = context().create_renderpass(
						{ attachment::declare_for(depthViews[i][cascade], on_load::clear, depth_stencil(), on_store::store) },
						[](avk::renderpass_sync& aRpSync) {
							if (aRpSync.is_external_pre_sync()) {
								aRpSync.mSourceStage = avk::pipeline_stage::top_of_pipe;
								aRpSync.mSourceMemoryDependency = {};
								aRpSync.mDestinationStage = avk::pipeline_stage::early_fragment_tests;
								aRpSync.mDestinationMemoryDependency = avk::memory_access::depth_stencil_attachment_write_access;
							}
							if (aRpSync.is_external_post_sync()) {
								// The next pipeline must wait before this pipeline has finished writing its color attachment output
								aRpSync.mSourceStage = avk::pipeline_stage::late_fragment_tests;
								aRpSync.mDestinationStage = avk::pipeline_stage::fragment_shader;
								// It's the same memory access for both, that needs to be synchronized:
								aRpSync.mSourceMemoryDependency = avk::memory_access::depth_stencil_attachment_write_access;
								aRpSync.mDestinationMemoryDependency = avk::memory_access::shader_buffers_and_images_read_access;
							}
						}
					);
					mShadowmapRenderpass.enable_shared_ownership();
				}

				// FIXME - filter, border
				depthViews[i][cascade].enable_shared_ownership();
				//mShadowmapImageSamplers[i].push_back(
				(mShadowmapImageSamplers[i])[cascade] =
					context().create_image_sampler(
						avk::shared(depthViews[i][cascade]),
						//context().create_sampler(avk::filter_mode::bilinear, avk::border_handling_mode::clamp_to_border, 0.f, [](avk::sampler_t & smp) { smp.config().setBorderColor(vk::BorderColor::eFloatOpaqueWhite); })
						context().create_sampler(avk::filter_mode::bilinear, avk::border_handling_mode::clamp_to_edge, 0.f,
							[](avk::sampler_t & smp) {
								smp.config().setCompareEnable(VK_TRUE).setCompareOp(vk::CompareOp::eLess);
							}
						)
					);
				
				mShadowmapPerCascade[cascade].mShadowmapFramebuffer[i] = context().create_framebuffer(avk::shared(mShadowmapRenderpass), avk::shared(depthViews[i][cascade]));
				mShadowmapPerCascade[cascade].mShadowmapFramebuffer[i]->initialize_attachments(sync::wait_idle(true));
			}
		}

		static bool first_time_here = true;
		if (first_time_here) {
			// fill in image samplers for (so far) unused cascades (need those valid for pipeline creation); first-time init only
			for (int cascade = mShadowMap.numCascades; cascade < SHADOWMAP_MAX_CASCADES; ++cascade) {
				for (decltype(numFif) i = 0; i < numFif; ++i) {
					if (!(mShadowmapImageSamplers[i])[cascade].has_value()) {
						(mShadowmapImageSamplers[i])[cascade] =
							context().create_image_sampler(
								avk::shared(depthViews[i][0]), // set to cascade #0
								context().create_sampler(avk::filter_mode::bilinear, avk::border_handling_mode::clamp_to_edge, 0.f,
									[](avk::sampler_t & smp) {
										smp.config().setCompareEnable(VK_TRUE).setCompareOp(vk::CompareOp::eLess);
									}
								)
							);
					}
				}
			}
		}
		first_time_here = false;

		// pipeline is created in prepare_common_pipelines
#endif
	}

	void re_init_shadowmap() {
		// if the user selected more cascades than were originally set, we may need to alloc more framebuffers
		if (mShadowMap.desiredNumCascades > mShadowMap.numCascades) {
			gvk::context().device().waitIdle();
			mShadowMap.numCascades = mShadowMap.desiredNumCascades;
			prepare_shadowmap();
		} else {
			mShadowMap.numCascades = mShadowMap.desiredNumCascades;
		}

		mShadowMap.shadowMapUtil.init(mSceneData.mBoundingBox, mQuakeCam.near_plane_distance(), mQuakeCam.far_plane_distance(), SHADOWMAP_SIZE, mShadowMap.numCascades, mShadowMap.autoCalcCascadeEnds);

	}

	// ac: output info about the scene structure
	void print_scene_debug_info(avk::resource_reference<gvk::orca_scene_t> aScene, bool withMeshNames = false)
	{
		auto &models = aScene->models();
		std::cout << "Scene debug info" << std::endl;
		std::cout << "#models " << models.size() << std::endl;
		for (int iModel = 0; iModel < models.size(); iModel++) {
			auto assimpScene = models[iModel].mLoadedModel->handle();
			std::cout << "Model #" << iModel << ", (" << assimpScene->mNumMeshes << " meshes):" << std::endl;
			int meshCount = 0;
			lukeMeshWalker(assimpScene, assimpScene->mRootNode,
				[&withMeshNames, &meshCount](std::string aFullName, unsigned int aMeshcount, std::string aMeshNames, aiNode *node)->bool {
					std::cout << aFullName << ":\t" << aMeshcount << " meshes:";
					for (size_t i = 0; i < node->mNumMeshes; i++) std::cout << " " << node->mMeshes[i];
					if (withMeshNames) std::cout << " (" << aMeshNames << ")";
					std::cout << std::endl;
					meshCount += node->mNumMeshes;
					return true;
				}
			);
			std::cout << "Total # meshes in scene graph:" << meshCount << std::endl;
		}
	}

	bool lukeMeshWalker(
		const aiScene *aiscene, aiNode *node,
		std::function<bool(std::string, unsigned int, std::string, aiNode*)> callback =
			[](std::string aFullName, unsigned int aMeshcount, std::string aMeshes, aiNode *node) {
				return true;
			}
	) {
		// passed function shall return true to continue or false to abort
		std::string fullName = node->mName.C_Str();
		aiNode *n = node->mParent;
		while (n) {
			fullName.insert(0, "/");
			fullName.insert(0, n->mName.C_Str());
			n = n ->mParent;
		}
		std::string meshNames;
		for (size_t i = 0; i < node->mNumMeshes; i++) {
			if (meshNames.length()) meshNames.append(",");
			meshNames.append("\"");
			meshNames.append(aiscene->mMeshes[node->mMeshes[i]]->mName.C_Str());
			meshNames.append("\"");
		}

		if (!callback(fullName, node->mNumMeshes, meshNames, node)) return false; // stop recursion

		for (size_t iChild = 0; iChild < node->mNumChildren; iChild++) {
			if (!lukeMeshWalker(aiscene, node->mChildren[iChild], callback)) return false; // stop recursion
		}
		return true; // continue recursion
	}

	uint32_t colToUint32(glm::vec4 col) {
		glm::ivec4 c(col * 255.f);
		return ((c.r & 0xff) << 24) | ((c.g & 0xff) << 16) | ((c.b & 0xff) << 8) | (c.a & 0xff);
	}

	std::string baseName(std::string path) {
		return path.substr(path.find_last_of("/\\") + 1);
	}

	// ac: print material properties
	void print_material_debug_info(avk::resource_reference<gvk::orca_scene_t> aScene) {
		auto distinctMaterialsOrca = aScene->distinct_material_configs_for_all_models(true);
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

	SceneType detect_scene_type(avk::resource_reference<gvk::orca_scene_t> aScene) {
		SceneType typ = SceneType::Unknown;

		auto &models = aScene->models();
		for (int iModel = 0; iModel < models.size(); iModel++) {
			auto assimpScene = models[iModel].mLoadedModel->handle();
			lukeMeshWalker(assimpScene, assimpScene->mRootNode,
				[&typ](std::string aFullName, unsigned int aMeshcount, std::string aMeshNames, aiNode *node)->bool {
					if (std::string::npos != aFullName.find("/Sheleg/")) {
						typ = SceneType::EmeraldSquare;
					} else if (std::string::npos != aFullName.find("/BistroExterior/")) {
						typ = SceneType::BistroExterior;
					} else if (std::string::npos != aFullName.find("sponza_structure.obj")) {
						typ = SceneType::Sponza;
					}
					return typ == SceneType::Unknown;
				}
			);
		}

		std::cout << "Detected scene: ";
		switch (typ) {
		case SceneType::EmeraldSquare:	std::cout << "Emerald Square"	<< std::endl; break;
		case SceneType::Sponza:			std::cout << "Sponza"			<< std::endl; break;
		case SceneType::BistroExterior:	std::cout << "Bistro exterior"	<< std::endl; break;
		default:						std::cout << "Unknown"			<< std::endl; break;
		}

		return typ;
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
		if (mSceneType == SceneType::EmeraldSquare) {
			res |= (std::string::npos != mat.mName.find(".DoubleSided"));
		}

		// In Sponza with plants: materials "leaf" and "Material__57" (vase plants)
		if (mSceneType == SceneType::Sponza) {
			res |= mat.mName == "leaf";
			res |= mat.mName == "Material__57";
		}

		if (mSceneType == SceneType::BistroExterior) {
			res |= mat.mName == "MASTER_Curtains";
			res |= mat.mName == "MASTER_Glass_Dirty_MASKED";
			res |= mat.mName == "MASTER_Forge_Metal";
			// TODO... more?
		}

		return res;
	}

	bool is_material_twosided(const gvk::material_config &mat) {
		// also mark all transparents as two sided (e.g. plants in sponza)
		return mat.mTwosided || has_material_transparency(mat);	// Emerald-Square leaves are not marked twosided, but can be found by name
	}

	void load_and_prepare_scene() // up to the point where all draw call data and material data has been assembled
	{
		using namespace avk;
		using namespace gvk;

		double t0 = glfwGetTime();
		std::cout << "Loading scene..." << std::endl;

		// Load a scene (in ORCA format) from file:
		auto scene = orca_scene_t::load_from_file(mSceneFileName, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace | (mFlipUvWithAssimp ? aiProcess_FlipUVs : 0) );

		mSceneType = detect_scene_type(scene);

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
#if ENABLE_RAYTRACING
		//assert(cgb::settings::gEnableBufferDeviceAddress);
		bufferUsageFlags |= vk::BufferUsageFlagBits::eShaderDeviceAddressKHR;	// TODO: need this for raytracing ?
#endif

		std::cout << "Parsing scene\r"; std::cout.flush();


		int counter = 0, counterlimit = 0; // 200;

		// Get all the different materials from the whole scene:
		auto distinctMaterialsOrca = scene->distinct_material_configs_for_all_models();
		std::vector<material_config> distinctMaterialConfigs;

		mSceneData.mMaxTransparentMeshgroups = mSceneData.mMaxOpaqueMeshgroups = 0;

		// for cache efficiency, we want to render meshgroups using the same material in sequence, so: walk the materials, find matching meshes, build meshgroup
		for (const auto& pair : distinctMaterialsOrca) {
			const int materialIndex = static_cast<int>(distinctMaterialConfigs.size());
			distinctMaterialConfigs.push_back(pair.first);
			assert (static_cast<size_t>(materialIndex + 1) == distinctMaterialConfigs.size());

			bool materialHasTransparency = has_material_transparency(pair.first);
			bool materialIsTwoSided      = pair.first.mCustomData[3] > 0;

			// walk the meshdefs having the current material (over ALL orca-models)
			for (const auto& modelAndMeshIndices : pair.second) {
				// Gather the model reference and the mesh indices of the same material in a vector:
				auto& modelData = scene->model_at_index(modelAndMeshIndices.mModelIndex);
				std::vector<std::tuple<resource_reference<const model_t>, std::vector<size_t>>> modelRefAndMeshIndices = { std::make_tuple(const_referenced(modelData.mLoadedModel), modelAndMeshIndices.mMeshIndices) };
				helpers::exclude_a_curtain(modelRefAndMeshIndices);

				if (modelRefAndMeshIndices.empty()) continue;

				// walk the individual meshes (actually these correspond to "mesh groups", referring to the same meshId) in the same-material-group
				for (auto& modelRefMeshIndicesPair : modelRefAndMeshIndices) {
					for (auto meshIndex : std::get<std::vector<size_t>>(modelRefMeshIndicesPair)) {
						counter++;
						std::cout << "Parsing scene " << counter << "\r"; std::cout.flush();

						std::vector<size_t> tmpMeshIndexVector = { meshIndex };
						std::vector<std::tuple<resource_reference<const model_t>, std::vector<size_t>>> selection = { std::make_tuple(const_referenced(modelData.mLoadedModel), tmpMeshIndexVector) };

						// build a meshgroup, covering ALL orca-instances
						// TODO: this way of buffer-building is not very efficient; better build buffers for ALL meshes in one go...

						// get the data of the current orca-mesh(group)
						auto [positions, indices] = get_vertices_and_indices(selection);
						auto texCoords            = mFlipManually ? get_2d_texture_coordinates_flipped(selection, 0) : get_2d_texture_coordinates(selection, 0);
						auto normals              = get_normals   (selection);
						auto tangents             = get_tangents  (selection);
						auto bitangents           = get_bitangents(selection);

						Meshgroup mg;
						mg.numIndices      = static_cast<uint32_t>(indices.size());
						mg.numVertices     = static_cast<uint32_t>(positions.size());
						mg.baseIndex       = static_cast<uint32_t>(mSceneData.mIndices.size());
						mg.baseVertex      = static_cast<uint32_t>(mSceneData.mPositions.size());
						mg.materialIndex   = materialIndex;
						mg.hasTransparency = materialHasTransparency;
						mg.isTwoSided	   = materialIsTwoSided;
						// for debugging:
						mg.orcaModelId = static_cast<uint32_t>(modelAndMeshIndices.mModelIndex);
						mg.orcaMeshId  = static_cast<uint32_t>(meshIndex);

						// bounding box (without transformations)
						mg.boundingBox_untransformed.calcFromPoints(positions.size(), positions.data());

						// append the data of the current mesh(group) to the scene vectors
						append_indices_and_vertex_data(
							additional_index_data (mSceneData.mIndices,		[&]() { return indices;	 }),
							additional_vertex_data(mSceneData.mPositions,	[&]() { return positions; })
						);
						insert_into(mSceneData.mTexCoords,  texCoords);
						insert_into(mSceneData.mNormals,    normals);
						insert_into(mSceneData.mTangents,   tangents);
						insert_into(mSceneData.mBitangents, bitangents);

						if (mg.hasTransparency) mSceneData.mMaxTransparentMeshgroups++; else mSceneData.mMaxOpaqueMeshgroups++;

						// collect all the instances of the meshgroup
						auto in_instance_transforms = get_mesh_instance_transforms(modelData.mLoadedModel, static_cast<int>(meshIndex), glm::mat4(1));
						// for each orca-instance of the loaded model, apply the instance transform and store the final transforms in the meshgroup
						for (size_t i = 0; i < modelData.mInstances.size(); ++i) {
							auto baseTransform = matrix_from_transforms(modelData.mInstances[i].mTranslation, glm::quat(modelData.mInstances[i].mRotation), modelData.mInstances[i].mScaling);
							for (auto t = 0; t < in_instance_transforms.size(); ++t) {
								MeshgroupPerInstanceData pid;
								pid.modelMatrix = baseTransform * in_instance_transforms[t];
								mg.perInstanceData.push_back(pid);
							}
						}

						#if ENABLE_RAYTRACING
							// temp data for ray tracing - FIXME
							mg.rayTracingTmp.rtIndexData      = indices;
							mg.rayTracingTmp.rtPositionData   = positions;
							mg.rayTracingTmp.rtTexCoordData   = texCoords;
							mg.rayTracingTmp.rtNormalsData    = normals;
							mg.rayTracingTmp.rtTangentsData   = tangents;
							mg.rayTracingTmp.rtBitangentsData = bitangents;
						#endif			

						mSceneData.mMeshgroups.push_back(mg);
					}
				}
			}
		}
		std::cout << std::endl;

		// count total # instances
		mSceneData.mNumTotalInstances = 0;
		for (auto &mg : mSceneData.mMeshgroups) mSceneData.mNumTotalInstances += mg.perInstanceData.size();

		// sort meshgroups by transparency (we want to render opaque objects first)
		std::sort(mSceneData.mMeshgroups.begin(), mSceneData.mMeshgroups.end(), [](Meshgroup a, Meshgroup b) { return static_cast<int>(a.hasTransparency) < static_cast<int>(b.hasTransparency); });

		// calc scene bounding box
		if (mSceneData.mMeshgroups.size()) {
			mSceneData.mBoundingBox.min = glm::vec3(std::numeric_limits<float>::max());
			mSceneData.mBoundingBox.max = glm::vec3(std::numeric_limits<float>::min());
			for (auto &mg : mSceneData.mMeshgroups) {
				for (auto &inst : mg.perInstanceData) {
					mSceneData.mBoundingBox.combineWith(glm::vec3(inst.modelMatrix * glm::vec4(mg.boundingBox_untransformed.min, 1.f)));
					mSceneData.mBoundingBox.combineWith(glm::vec3(inst.modelMatrix * glm::vec4(mg.boundingBox_untransformed.max, 1.f)));
				}
			}
		}

		// create all the buffers - for details, see upload_materials_and_vertex_data_to_gpu()
		size_t numMeshgroups = mSceneData.mMeshgroups.size();
		size_t numInstances  = mSceneData.mNumTotalInstances;

		mSceneData.mIndexBuffer              = context().create_buffer(memory_usage::device, {}, index_buffer_meta::create_from_data(mSceneData.mIndices));
		mSceneData.mPositionsBuffer          = context().create_buffer(memory_usage::device, {}, vertex_buffer_meta::create_from_data(mSceneData.mPositions).describe_only_member(mSceneData.mPositions[0], content_description::position));
		mSceneData.mTexCoordsBuffer          = context().create_buffer(memory_usage::device, {}, vertex_buffer_meta::create_from_data(mSceneData.mTexCoords));
		mSceneData.mNormalsBuffer            = context().create_buffer(memory_usage::device, {}, vertex_buffer_meta::create_from_data(mSceneData.mNormals));
		mSceneData.mTangentsBuffer           = context().create_buffer(memory_usage::device, {}, vertex_buffer_meta::create_from_data(mSceneData.mTangents));
		mSceneData.mBitangentsBuffer         = context().create_buffer(memory_usage::device, {}, vertex_buffer_meta::create_from_data(mSceneData.mBitangents));
		mSceneData.mAttributesBuffer         = context().create_buffer(memory_usage::device, {}, storage_buffer_meta::create_from_size(numInstances  * sizeof(MeshgroupPerInstanceData)));
		mSceneData.mCullingBoundingBoxBuffer = context().create_buffer(memory_usage::device, {}, storage_buffer_meta::create_from_size(numInstances  * sizeof(CullingBoundingBox)));
		mSceneData.mMeshgroupInfoBuffer      = context().create_buffer(memory_usage::device, {}, storage_buffer_meta::create_from_size(numMeshgroups * sizeof(MeshgroupBasicInfoGpu)));

		rdoc::labelBuffer(mSceneData.mIndexBuffer             ->handle(), "scene_IndexBuffer");
		rdoc::labelBuffer(mSceneData.mPositionsBuffer         ->handle(), "scene_PositionsBuffer");
		rdoc::labelBuffer(mSceneData.mTexCoordsBuffer         ->handle(), "scene_TexCoordsBuffer");
		rdoc::labelBuffer(mSceneData.mNormalsBuffer           ->handle(), "scene_NormalsBuffer");
		rdoc::labelBuffer(mSceneData.mTangentsBuffer          ->handle(), "scene_TangentsBuffer");
		rdoc::labelBuffer(mSceneData.mBitangentsBuffer        ->handle(), "scene_BitangentsBuffer");
		rdoc::labelBuffer(mSceneData.mAttributesBuffer        ->handle(), "scene_AttributesBuffer");
		rdoc::labelBuffer(mSceneData.mCullingBoundingBoxBuffer->handle(), "scene_CullingBoundingBoxBuffer");
		rdoc::labelBuffer(mSceneData.mMeshgroupInfoBuffer     ->handle(), "scene_MeshgroupInfoBuffer");

		auto numFif = context().main_window()->number_of_frames_in_flight();
		memory_usage memoryUsage = (SCENE_DATA_BUFFER_ON_DEVICE ? memory_usage::device : memory_usage::host_coherent);
		auto indirect = vk::BufferUsageFlagBits::eIndirectBuffer;
		for (decltype(numFif) i = 0; i < numFif; ++i) {
			mSceneData.mDrawnMeshgroupBuffer         [i] = context().create_buffer(memoryUsage,                 {},         storage_buffer_meta::create_from_size(numMeshgroups * sizeof(DrawnMeshgroupData)));
			mSceneData.mDrawnMeshAttribIndexBuffer   [i] = context().create_buffer(memoryUsage,                 {},         storage_buffer_meta::create_from_size(numInstances  * sizeof(uint32_t)));
			mSceneData.mMeshgroupsLayoutInfoBuffer   [i] = context().create_buffer(memoryUsage,                 {},         storage_buffer_meta::create_from_size(sizeof(uint32_t)));
			mSceneData.mDrawCommandsBuffer           [i] = context().create_buffer(memoryUsage,                 {indirect}, storage_buffer_meta::create_from_size(numMeshgroups * sizeof(vk::DrawIndexedIndirectCommand)));
			mSceneData.mDrawCountBuffer              [i] = context().create_buffer(memoryUsage,                 {indirect}, storage_buffer_meta::create_from_size(2             * sizeof(uint32_t)));
			mSceneData.mCullingUniformsBuffer        [i] = context().create_buffer(memory_usage::host_coherent, {},         uniform_buffer_meta::create_from_size(sizeof(CullingUniforms)));
			mSceneData.mCullingVisibilityBuffer      [i] = context().create_buffer(memoryUsage,                 {},         storage_buffer_meta::create_from_size(numInstances  * sizeof(uint32_t)));

			rdoc::labelBuffer(mSceneData.mDrawnMeshgroupBuffer         [i]->handle(), "scene_DrawnMeshgroupBuffer", i);
			rdoc::labelBuffer(mSceneData.mDrawnMeshAttribIndexBuffer   [i]->handle(), "scene_DrawnMeshAttribIndexBuffer", i);
			rdoc::labelBuffer(mSceneData.mMeshgroupsLayoutInfoBuffer   [i]->handle(), "scene_MeshgroupsLayoutInfoBuffer", i);
			rdoc::labelBuffer(mSceneData.mDrawCommandsBuffer           [i]->handle(), "scene_DrawCommandsBuffer", i);
			rdoc::labelBuffer(mSceneData.mDrawCountBuffer              [i]->handle(), "scene_DrawCountBuffer", i);
			rdoc::labelBuffer(mSceneData.mCullingUniformsBuffer        [i]->handle(), "scene_CullingUniformsBuffer", i);
			rdoc::labelBuffer(mSceneData.mCullingVisibilityBuffer      [i]->handle(), "scene_CullingVisibilityBuffer", i);
		}

		mSceneData.print_stats();

		double tParse = glfwGetTime();

		// load and add moving objects
		std::cout << "Loading extra models" << std::endl;
		size_t totalNumBoneMatrices = 0;
		mMovingObjectFirstMatIdx = static_cast<int>(distinctMaterialConfigs.size());
		for (size_t iMover = 0; iMover < mMovingObjectDefs.size(); iMover++)
		{
			// FIXME - this only works for objects with 1 mesh (at least only the first mesh is rendered)
			auto objdef = mMovingObjectDefs[iMover];
			if (!std::filesystem::exists(objdef.filename)) {
				LOG_WARNING("Object file \"" + std::string(objdef.filename) + "\" does not exist - falling back to default sphere)");
				mMovingObjectDefs[iMover].name = "Not available"; // (std::string("N/A (") + objdef.name + ")").c_str();
				objdef = mMovingObjectDefs[0];
			}
			std::cout << "Loading model \"" << objdef.name << "\" ..." << std::endl;
			auto model = model_t::load_from_file(objdef.filename, aiProcess_Triangulate | aiProcess_CalcTangentSpace);

			auto &dynObj = mDynObjects.emplace_back(dynamic_object{});
			dynObj.mIsAnimated = objdef.animId >= 0;
			dynObj.mBaseTransform = objdef.modelMatrix;

			// iterate through all the model's meshes (TODO: combine materials first? ( model->distinct_material_configs() ) - probably should do that in a real-life app to avoid loading duplicate textures etc...)
			for (auto meshIndex = 0; meshIndex < model->num_meshes(); ++meshIndex) {
				bool didWarnStrangeBoneWeights = false;

				auto &meshData = dynObj.mMeshData.emplace_back(dynamic_object_mesh_data{});

				// get material index in our global material buffer, modify material and push it to the global buffer
				auto material = model->material_config_for_mesh(meshIndex);
				material.mCustomData[0] = iMover + 1.f; // moving object id + 1

				if (material.mAmbientReflectivity == glm::vec4(0)) material.mAmbientReflectivity = glm::vec4(glm::vec3(0.1f), 0); // give it some ambient reflectivity if it has none

				int texCoordSet = 0;

				// fix goblin model - has no texture assigned, specular reflectivity is through the roof
				if (objdef.name == "Goblin" && meshIndex == 0) {
					material.mDiffuseTex = "assets/goblin.dds";
					material.mSpecularReflectivity = glm::vec4(0);
				}

				// fix dragon model
				if (objdef.name == "Dragon") {
					material.mDiffuseTex		= "assets/optional/dragon/Dragon_ground_color.jpg";
					material.mSpecularTex		= "";
					material.mAmbientTex		= "";
					material.mEmissiveTex		= "";
					material.mHeightTex			= "";
					material.mNormalsTex		= "assets/optional/dragon/Dragon_Nor.jpg";
					material.mShininessTex		= "";
					material.mOpacityTex		= "";
					material.mDisplacementTex	= "";
					material.mReflectionTex		= "";
					material.mLightmapTex		= "";
					material.mExtraTex			= "";
					texCoordSet = 1;	// 0 = for bump mapping?
				}

				meshData.mMaterialIndex = static_cast<int>(distinctMaterialConfigs.size());
				distinctMaterialConfigs.push_back(material);


				// get the mesh data from the loaded model
				auto selection = make_models_and_meshes_selection(model, meshIndex);
				auto [vertices, indices] = get_vertices_and_indices(selection);
				auto texCoords = mFlipManually ? get_2d_texture_coordinates_flipped(selection, texCoordSet) : get_2d_texture_coordinates(selection, texCoordSet);
				auto normals	= get_normals(selection);
				auto tangents	= get_tangents(selection);
				auto bitangents	= get_bitangents(selection);

				std::vector<glm::vec4>  boneWeights;
				std::vector<glm::uvec4> boneIndices;
				if (dynObj.mIsAnimated) {
					boneWeights = get_bone_weights(selection, true /* normalize */);
					boneIndices = get_bone_indices(selection);
				}

				// Create all the GPU buffers, but don't fill yet:
				meshData.mIndexBuffer			= context().create_buffer(memory_usage::device, bufferUsageFlags, index_buffer_meta::        create_from_data(indices),
																												  uniform_texel_buffer_meta::create_from_data(indices).set_format<glm::uvec3>());
				meshData.mPositionsBuffer		= context().create_buffer(memory_usage::device, bufferUsageFlags, vertex_buffer_meta::       create_from_data(vertices).describe_only_member(vertices[0], content_description::position),
																												  uniform_texel_buffer_meta::create_from_data(vertices).describe_only_member(vertices[0], content_description::position));
				meshData.mTexCoordsBuffer		= context().create_buffer(memory_usage::device, bufferUsageFlags, vertex_buffer_meta::create_from_data(texCoords));
				meshData.mNormalsBuffer			= context().create_buffer(memory_usage::device, bufferUsageFlags, vertex_buffer_meta::create_from_data(normals));
				meshData.mTangentsBuffer		= context().create_buffer(memory_usage::device, bufferUsageFlags, vertex_buffer_meta::create_from_data(tangents));
				meshData.mBitangentsBuffer		= context().create_buffer(memory_usage::device, bufferUsageFlags, vertex_buffer_meta::create_from_data(bitangents));
				if (dynObj.mIsAnimated) {
					meshData.mBoneWeightsBuffer	= context().create_buffer(memory_usage::device, bufferUsageFlags, vertex_buffer_meta::create_from_data(boneWeights));
					meshData.mBoneIndicesBuffer	= context().create_buffer(memory_usage::device, bufferUsageFlags, vertex_buffer_meta::create_from_data(boneIndices));
				}

				// store mesh data for later upload
				meshData.mIndices		= std::move(indices);
				meshData.mPositions		= std::move(vertices);
				meshData.mTexCoords		= std::move(texCoords);
				meshData.mNormals		= std::move(normals);
				meshData.mTangents		= std::move(tangents);
				meshData.mBitangents	= std::move(bitangents);
				meshData.mBoneWeights	= std::move(boneWeights);
				meshData.mBoneIndices	= std::move(boneIndices);

				// store the different instance-transforms of the mesh (aka "parts" of the model)
				auto transforms = get_mesh_instance_transforms(model, static_cast<int>(meshIndex), glm::mat4(1)); // TODO <- apply objdef.modelMatrix here or keep separate in dynObj ?
				for (auto &tform : transforms) {
					dynObj.mParts.emplace_back(dynamic_object_part{static_cast<int>(meshIndex), tform});
				}
			}

			// load animation
			dynObj.mActiveAnimation = 0;
			if (dynObj.mIsAnimated) {
				//static bool hadAnim = false;
				//if (hadAnim) { throw runtime_error("Not more than 1 anim for now!"); }	// FIXME later! for now, only 1 animated object is supported
				//hadAnim = true;

				auto allMeshIndices = model->select_all_meshes();

				for (auto meshIdx : allMeshIndices) {
					if (model->num_bone_matrices(meshIdx) > MAX_BONES) {
						throw avk::runtime_error("Model mesh #" + std::to_string(meshIdx) + " has more bones (" + std::to_string(model->num_bone_matrices(meshIdx)) + ") than MAX_BONES (" + std::to_string(MAX_BONES) + ") !");
					}
				}

				auto anim_clip = model->load_animation_clip(objdef.animId, 0, 10'000'000);
				//printf("Loaded anim clip, start ticks=%f, end ticks=%f, tps=%f\n", anim_clip.mStartTicks, anim_clip.mEndTicks, anim_clip.mTicksPerSecond);
				assert(anim_clip.mTicksPerSecond > 0.0 && anim_clip.mEndTicks > anim_clip.mStartTicks);

				dynObj.mBoneMatrices.resize(MAX_BONES * allMeshIndices.size(), glm::mat4(0));
				totalNumBoneMatrices += dynObj.mBoneMatrices.size();
				dynObj.mAnimations.push_back(model->prepare_animation(objdef.animId, allMeshIndices));
				dynObj.mAnimClips.push_back(anim_clip);
			}
		}
		prepare_bone_matrices_buffer(totalNumBoneMatrices);
		// --end moving objects

		// load test images
		mTestImages.clear();
		for (auto &imgDef : mImageDefs) {
			mTestImages.push_back(context().create_image_view(create_image_from_file(imgDef.filename, true, true, false /* no flip */)));
		}
		mTestImageSampler_bilinear = context().create_sampler(filter_mode::bilinear,         border_handling_mode::clamp_to_edge);
		mTestImageSampler_nearest  = context().create_sampler(filter_mode::nearest_neighbor, border_handling_mode::clamp_to_edge);


		std::cout << "Loading textures... "; std::cout.flush();
		// Convert the material configs (that were gathered above) into a GPU-compatible format:
		// "GPU-compatible format" in this sense means that we'll get two things out of the call to `convert_for_gpu_usage`:
		//   1) Material data in a properly aligned format, suitable for being uploaded into a GPU buffer (but not uploaded into a buffer yet!)
		//   2) Image samplers (which contain images and samplers) of all the used textures, already uploaded to the GPU.
		std::tie(mMaterialData, mImageSamplers) = convert_for_gpu_usage(
			distinctMaterialConfigs, false,
			mFlipTexturesInLoader,
			mDisableMip ? image_usage::general_image : image_usage::general_texture,
			[](){ return to_filter_mode(context().physical_device().getProperties().limits.maxSamplerAnisotropy, true); }(), // set to max. anisotropy
			//avk::filter_mode::bilinear, // JUST TESTING manual lod calculation - REMOVEME!!
			border_handling_mode::repeat,
			sync::wait_idle(true)
		);
		std::cout << "done" << std::endl; 

		// Create a GPU buffer that will get filled with the material data:
		mMaterialBuffer = context().create_buffer(
			memory_usage::device, {},
			storage_buffer_meta::create_from_data(mMaterialData)
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
			int idxDir  = helpers::get_lightsource_type_begin_index(lightsource_type::directional);
			int idxDir2 = helpers::get_lightsource_type_end_index(lightsource_type::directional);
			int idxAmb  = helpers::get_lightsource_type_begin_index(lightsource_type::ambient);
			int idxAmb2 = helpers::get_lightsource_type_end_index(lightsource_type::ambient);
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

#if ENABLE_RAYTRACING
	void init_raytracing() {
		using namespace avk;
		using namespace gvk;

		// this is heavily based on the gears-vk raytracing example

		vk::BufferUsageFlags bufferUsage = { vk::BufferUsageFlagBits::eShaderDeviceAddressKHR };

		// create temp buffers for indices, vertices - FIXME!
		std::vector<uint32_t> materialIndices;
		for (auto &mg : mSceneData.mMeshgroups) {
			mg.rayTracingTmp.rtIndexBuffer		= context().create_buffer(memory_usage::device, bufferUsage,
																		  index_buffer_meta::        create_from_data(mg.rayTracingTmp.rtIndexData),
																		  uniform_texel_buffer_meta::create_from_data(mg.rayTracingTmp.rtIndexData).set_format<glm::uvec3>());
			mg.rayTracingTmp.rtPositionBuffer	= context().create_buffer(memory_usage::device, bufferUsage,
																		  vertex_buffer_meta::       create_from_data(mg.rayTracingTmp.rtPositionData).describe_only_member(mg.rayTracingTmp.rtPositionData[0], content_description::position),
																		  uniform_texel_buffer_meta::create_from_data(mg.rayTracingTmp.rtPositionData).describe_only_member(mg.rayTracingTmp.rtPositionData[0], content_description::position));
			mg.rayTracingTmp.rtIndexBuffer.enable_shared_ownership();
			mg.rayTracingTmp.rtPositionBuffer.enable_shared_ownership();
			mg.rayTracingTmp.rtIndexBuffer		->fill(mg.rayTracingTmp.rtIndexData.data(),    0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler())); // FIXME - sync ok?
			mg.rayTracingTmp.rtPositionBuffer	->fill(mg.rayTracingTmp.rtPositionData.data(), 0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler())); // FIXME - sync ok?

			// add buffers to arrays of buffer-views (to be indexed via meshgroup id in the shader)
			mRtIndexBuffersArray.push_back( context().create_buffer_view(shared(mg.rayTracingTmp.rtIndexBuffer)) );

			// add texture coords in the same way
			auto tbuf = context().create_buffer(memory_usage::device, bufferUsage, uniform_texel_buffer_meta::create_from_data(mg.rayTracingTmp.rtTexCoordData).describe_only_member(mg.rayTracingTmp.rtTexCoordData[0]));
			tbuf->fill(mg.rayTracingTmp.rtTexCoordData.data(),  0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler()));
			mRtTexCoordBuffersArray.push_back( context().create_buffer_view(owned(tbuf)) );

			// and normals, tangents, bitangents ...
			auto nbufN = context().create_buffer(memory_usage::device, bufferUsage, uniform_texel_buffer_meta::create_from_data(mg.rayTracingTmp.rtNormalsData   ).describe_only_member(mg.rayTracingTmp.rtNormalsData   [0]));
			auto nbufT = context().create_buffer(memory_usage::device, bufferUsage, uniform_texel_buffer_meta::create_from_data(mg.rayTracingTmp.rtTangentsData  ).describe_only_member(mg.rayTracingTmp.rtTangentsData  [0]));
			auto nbufB = context().create_buffer(memory_usage::device, bufferUsage, uniform_texel_buffer_meta::create_from_data(mg.rayTracingTmp.rtBitangentsData).describe_only_member(mg.rayTracingTmp.rtBitangentsData[0]));
			nbufN->fill(mg.rayTracingTmp.rtNormalsData   .data(),  0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler()));
			nbufT->fill(mg.rayTracingTmp.rtTangentsData  .data(),  0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler()));
			nbufB->fill(mg.rayTracingTmp.rtBitangentsData.data(),  0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler()));
			mRtNormalsBuffersArray   .push_back( context().create_buffer_view(owned(nbufN)) );
			mRtTangentsBuffersArray  .push_back( context().create_buffer_view(owned(nbufT)) );
			mRtBitangentsBuffersArray.push_back( context().create_buffer_view(owned(nbufB)) );

			// and positions too (for LOD approximation)
			mRtPositionsBuffersArray.push_back( context().create_buffer_view(shared(mg.rayTracingTmp.rtPositionBuffer)) );

			// and store material index
			materialIndices.push_back(mg.materialIndex);
		}

		// do the same for dynamic objects (and set their mRtCustomIndexBase property)
		uint32_t dynObj_nextCustomIndex = static_cast<uint32_t>(mSceneData.mMeshgroups.size());
		for (auto &dynObj : mDynObjects) {
			dynObj.mRtCustomIndexBase = dynObj_nextCustomIndex;
			dynObj_nextCustomIndex += static_cast<uint32_t>(dynObj.mMeshData.size());	// each *mesh* of the dynObj needs its own custom index (to index texcoords,normals,indexbufs etc.)
			assert(dynObj.mRtCustomIndexBase == mRtIndexBuffersArray.size());

			for (auto &mesh : dynObj.mMeshData) {
				mesh.mIndexBuffer.enable_shared_ownership();
				mesh.mPositionsBuffer.enable_shared_ownership();

				//auto ibuf = context().create_buffer(memory_usage::device, bufferUsage, uniform_texel_buffer_meta::create_from_data(mesh.mIndices).set_format<glm::uvec3>());
				//ibuf->fill(mesh.mIndices.data(),     0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler()));
				//mRtIndexBuffersArray.push_back( context().create_buffer_view(owned(ibuf)) );
				mRtIndexBuffersArray.push_back( context().create_buffer_view(shared(mesh.mIndexBuffer)) );

				auto tbuf = context().create_buffer(memory_usage::device, bufferUsage, uniform_texel_buffer_meta::create_from_data(mesh.mTexCoords).describe_only_member(mesh.mTexCoords[0]));
				tbuf->fill(mesh.mTexCoords.data(),  0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler()));
				mRtTexCoordBuffersArray.push_back( context().create_buffer_view(owned(tbuf)) );

				auto nbufN = context().create_buffer(memory_usage::device, bufferUsage, uniform_texel_buffer_meta::create_from_data(mesh.mNormals   ).describe_only_member(mesh.mNormals   [0]));
				auto nbufT = context().create_buffer(memory_usage::device, bufferUsage, uniform_texel_buffer_meta::create_from_data(mesh.mTangents  ).describe_only_member(mesh.mTangents  [0]));
				auto nbufB = context().create_buffer(memory_usage::device, bufferUsage, uniform_texel_buffer_meta::create_from_data(mesh.mBitangents).describe_only_member(mesh.mBitangents[0]));
				nbufN->fill(mesh.mNormals   .data(),  0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler()));
				nbufT->fill(mesh.mTangents  .data(),  0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler()));
				nbufB->fill(mesh.mBitangents.data(),  0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler()));
				mRtNormalsBuffersArray   .push_back( context().create_buffer_view(owned(nbufN)) );
				mRtTangentsBuffersArray  .push_back( context().create_buffer_view(owned(nbufT)) );
				mRtBitangentsBuffersArray.push_back( context().create_buffer_view(owned(nbufB)) );

				mRtPositionsBuffersArray.push_back( context().create_buffer_view(shared(mesh.mPositionsBuffer)) );

				materialIndices.push_back(mesh.mMaterialIndex);
			}
		}

		// create a buffer holding all material indices
		mRtMaterialIndexBuffer = context().create_buffer(memory_usage::device, bufferUsage, storage_buffer_meta::create_from_data(materialIndices));
		mRtMaterialIndexBuffer->fill(materialIndices.data(), 0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler())); // FIXME - sync ok?


		// build bottom level acceleration structures, one per meshgroup
		for (auto iMg = 0; iMg < mSceneData.mMeshgroups.size(); iMg++) {
			auto &mg = mSceneData.mMeshgroups[iMg];

			auto blas = context().create_bottom_level_acceleration_structure({ acceleration_structure_size_requirements::from_buffers(vertex_index_buffer_pair{mg.rayTracingTmp.rtPositionBuffer, mg.rayTracingTmp.rtIndexBuffer}) }, true);
			blas.enable_shared_ownership();
			mSceneData.mBLASs.push_back(blas);

			// generate a geometry instance for each instance of the mesh group
			for (auto &inst_data : mg.perInstanceData) {
				auto geoInstance = context().create_geometry_instance(blas);
				geoInstance.set_transform_column_major(to_array(inst_data.modelMatrix));
				geoInstance.set_custom_index(iMg);	// custom index = meshgroup id
				geoInstance.set_mask(mg.hasTransparency ? RAYTRACING_CULLMASK_TRANSPARENT : RAYTRACING_CULLMASK_OPAQUE);
				if (mg.hasTransparency) geoInstance.force_non_opaque();
				if (mg.isTwoSided)      geoInstance.disable_culling();
				mSceneData.mGeometryInstances.push_back(std::move(geoInstance));
				mSceneData.mDebugGeoInstTransforms.push_back(inst_data.modelMatrix);
			}

			mSceneData.mBLASs.back()->build({ vertex_index_buffer_pair{mg.rayTracingTmp.rtPositionBuffer, mg.rayTracingTmp.rtIndexBuffer} }, {},
				sync::with_barriers(context().main_window()->command_buffer_lifetime_handler())
				//avk::sync::with_barriers(
				//	[posBfr = mg.rayTracingTmp.rtPositionBuffer, idxBfr = mg.rayTracingTmp.rtIndexBuffer](avk::command_buffer cb) {
				//		cb->set_custom_deleter([lPosBfr = std::move(posBfr), lIdxBfr = std::move(idxBfr)](){});
				//		gvk::context().main_window()->handle_lifetime(avk::owned(cb));
				//	},
				//	{}, {}
				//)
			);
		}

		// ... and build BLASs for dynamic objects (but no geo instances yet)
		size_t maxGeometryInstancesForDynObjs = 0;
		for (int iDynObj = 0; iDynObj < mDynObjects.size(); ++iDynObj) {
			auto &dynObj = mDynObjects[iDynObj];
			for (int iMesh = 0; iMesh < dynObj.mMeshData.size(); ++iMesh) {
				auto &mesh = dynObj.mMeshData[iMesh];

#if 0
				// doesn't work - NSight says the corresponding vertex buffer in AS is all zero - WHY???

				mesh.mBLAS = context().create_bottom_level_acceleration_structure({ acceleration_structure_size_requirements::from_buffers(vertex_index_buffer_pair{mesh.mPositionsBuffer, mesh.mIndexBuffer}) }, true);
				mesh.mBLAS.enable_shared_ownership();

				mesh.mBLAS->build({ vertex_index_buffer_pair{mesh.mPositionsBuffer, mesh.mIndexBuffer} }, {},
					sync::with_barriers(context().main_window()->command_buffer_lifetime_handler())
					//avk::sync::with_barriers(
					//	[posBfr = mesh.mPositionsBuffer, idxBfr = mesh.mIndexBuffer](avk::command_buffer cb) {		// TODO!! : check if deleter is appropriate here! probably not!
					//		cb->set_custom_deleter([lPosBfr = std::move(posBfr), lIdxBfr = std::move(idxBfr)](){});
					//		gvk::context().main_window()->handle_lifetime(avk::owned(cb));
					//	},
					//	{}, {}
					//)
				);
#else

				// --- debug! this works
				auto tmpIndexBuffer		= context().create_buffer(memory_usage::device, bufferUsage, index_buffer_meta::create_from_data(mesh.mIndices));
				auto tmpPositionsBuffer	= context().create_buffer(memory_usage::device, bufferUsage, vertex_buffer_meta::create_from_data(mesh.mPositions).describe_only_member(mesh.mPositions[0], content_description::position));
				tmpIndexBuffer.enable_shared_ownership();
				tmpPositionsBuffer.enable_shared_ownership();
				tmpIndexBuffer		->fill(mesh.mIndices.data(),   0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler())); // FIXME - sync ok?
				tmpPositionsBuffer	->fill(mesh.mPositions.data(), 0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler())); // FIXME - sync ok?

				mesh.mBLAS = context().create_bottom_level_acceleration_structure({ acceleration_structure_size_requirements::from_buffers(vertex_index_buffer_pair{tmpPositionsBuffer, tmpIndexBuffer}) }, true);
				mesh.mBLAS.enable_shared_ownership();

				mesh.mBLAS->build({ vertex_index_buffer_pair{tmpPositionsBuffer, tmpIndexBuffer} }, {},
					//sync::with_barriers(context().main_window()->command_buffer_lifetime_handler())
					avk::sync::with_barriers(
						[posBfr = tmpPositionsBuffer, idxBfr = tmpIndexBuffer](avk::command_buffer cb) {		// TODO!! : check if deleter is appropriate here! probably not!
							cb->set_custom_deleter([lPosBfr = std::move(posBfr), lIdxBfr = std::move(idxBfr)](){});
							gvk::context().main_window()->handle_lifetime(avk::owned(cb));
						},
						{}, {}
					)
				);
				// ---
#endif
			}

			maxGeometryInstancesForDynObjs = std::max(maxGeometryInstancesForDynObjs, dynObj.mParts.size());
		}

		//// just for debugging - add first dynObj to static geo
		//auto &dynObj = mDynObjects[0];
		//for (auto &part : dynObj.mParts) {
		//	auto &mesh = dynObj.mMeshData[part.mMeshIndex];
		//	auto geoInstance = context().create_geometry_instance(mesh.mBLAS);
		//	geoInstance.set_transform_column_major(to_array(/*dynObj.mMovementMatrix_in_AS * */ dynObj.mBaseTransform * part.mMeshTransform)); // TODO: check! not for animated objs...
		//	geoInstance.set_custom_index(dynObj.mRtCustomIndexBase + part.mMeshIndex);
		//	mSceneData.mGeometryInstances.push_back(std::move(geoInstance));
		//}

		// build top level acceleration structure, one per frame
		mAllGeometryInstances = mSceneData.mGeometryInstances;
		auto numFif = context().main_window()->number_of_frames_in_flight();
		for (decltype(numFif) i = 0; i < numFif; ++i) {
			auto tlas = context().create_top_level_acceleration_structure(static_cast<uint32_t>(mSceneData.mGeometryInstances.size() + maxGeometryInstancesForDynObjs), true);

			tlas->build(mSceneData.mGeometryInstances, {}, avk::sync::with_barriers(
				gvk::context().main_window()->command_buffer_lifetime_handler(),
				// Sync before building the TLAS:
				[](avk::command_buffer_t& commandBuffer, avk::pipeline_stage destinationStage, std::optional<avk::read_memory_access> readAccess){
					assert(avk::pipeline_stage::acceleration_structure_build == destinationStage);
					assert(!readAccess.has_value() || avk::memory_access::acceleration_structure_read_access == readAccess.value().value());
					// Wait on all the BLAS builds that happened before (and their memory):
					commandBuffer.establish_global_memory_barrier_rw(
						avk::pipeline_stage::acceleration_structure_build, destinationStage,
						avk::memory_access::acceleration_structure_write_access, readAccess
					);
				},
				// Whatever comes after must wait for this TLAS build to finish (and its memory to be made available):
				//   However, that's exactly what the default_handler_after_operation
				//   does, so let's just use that (it is also the default value for
				//   this handler)
				avk::sync::presets::default_handler_after_operation
			));
			mSceneData.mTLASs[i] = std::move(tlas);
		}

		// create images to render into
		std::vector<avk::command_buffer> layoutTransitions;
		auto win = context().main_window();
		//const auto wdth = win->resolution().x;
		//const auto hght = win->resolution().y;
		//const auto frmt = format_from_window_color_buffer(win);
		const auto wdth = mLoResolution.x;
		const auto hght = mLoResolution.y;
		const auto frmt = IMAGE_FORMAT_RAYTRACE;
		for (decltype(numFif) i = 0; i < numFif; ++i) {
			auto offscreenImage = context().create_image(wdth, hght, frmt, 1, memory_usage::device, image_usage::general_storage_image);
			offscreenImage->transition_to_layout({}, sync::with_barriers(win->command_buffer_lifetime_handler()));
			mRtImageViews[i] = context().create_image_view(owned(offscreenImage));
			assert((mRtImageViews[i]->config().subresourceRange.aspectMask & vk::ImageAspectFlagBits::eColor) == vk::ImageAspectFlagBits::eColor);
		}
		{
			auto dummySegMask = context().create_image(1, 1, TAA_IMAGE_FORMAT_SEGMASK, 1, memory_usage::device, image_usage::general_storage_image);
			mRtDummySegMask = context().create_image_view(owned(dummySegMask));
			layoutTransitions.emplace_back(std::move(mRtDummySegMask->get_image().transition_to_layout({}, avk::sync::with_barriers_by_return({}, {})).value()));
		}
		std::vector<avk::resource_reference<avk::command_buffer_t>> commandBufferReferences;
		std::transform(std::begin(layoutTransitions), std::end(layoutTransitions), std::back_inserter(commandBufferReferences), [](avk::command_buffer& cb) { return avk::referenced(*cb); });
		auto fen = mQueue->submit_with_fence(commandBufferReferences);
		fen->wait_until_signalled();


		// create a halton sequence for sample offsets + a buffer for them
		auto halton = helpers::halton_2_3<RAYTRACING_MAX_SAMPLES_PER_PIXEL>(glm::vec2(1.f));
		//for (auto &v : halton) printf("%f %f\n", v.x, v.y);
		mRtPixelOffsetBuffer = context().create_buffer(memory_usage::device, bufferUsage, storage_buffer_meta::create_from_data(halton));
		mRtPixelOffsetBuffer->fill(halton.data(), 0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler())); // FIXME - sync ok?

		// create animated object normals and normals-offset buffers
		// first determine max size
		size_t maxAnimObjNormals = 0;
		size_t maxAnimObjMeshes  = 0;
		for (auto &dynObj : mDynObjects) {
			if (dynObj.mIsAnimated) {
				size_t cntNrm = 0;
				for (auto &mesh : dynObj.mMeshData) cntNrm += mesh.mNormals.size();
				maxAnimObjMeshes  = std::max(maxAnimObjMeshes, dynObj.mMeshData.size());
				maxAnimObjNormals = std::max(maxAnimObjNormals, cntNrm);
			}
		}
		std::vector<glm::vec3> dummyVec3s(maxAnimObjNormals);
		std::vector<uint32_t>  dummyNTBOff(maxAnimObjMeshes);
		for (decltype(numFif) i = 0; i < numFif; ++i) {
			mRtAnimObjNormalsBuffer       [i] = context().create_buffer(memory_usage::device, bufferUsage, uniform_texel_buffer_meta::create_from_data(dummyVec3s).set_format<glm::vec3>());
			mRtAnimObjTangentsBuffer      [i] = context().create_buffer(memory_usage::device, bufferUsage, uniform_texel_buffer_meta::create_from_data(dummyVec3s).set_format<glm::vec3>());
			mRtAnimObjBitangentsBuffer    [i] = context().create_buffer(memory_usage::device, bufferUsage, uniform_texel_buffer_meta::create_from_data(dummyVec3s).set_format<glm::vec3>());
			mRtAnimObjPositionsBuffer     [i] = context().create_buffer(memory_usage::device, bufferUsage, uniform_texel_buffer_meta::create_from_data(dummyVec3s).set_format<glm::vec3>());
			mRtAnimObjNTBOffsetBuffer     [i] = context().create_buffer(memory_usage::device, bufferUsage, storage_buffer_meta::create_from_data(dummyNTBOff));
			mRtAnimObjNormalsBuffer       [i]->fill(dummyVec3s.data(),  0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler())); // filling isn't really necessary, is it?
			mRtAnimObjTangentsBuffer      [i]->fill(dummyVec3s.data(),  0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler())); // filling isn't really necessary, is it?
			mRtAnimObjBitangentsBuffer    [i]->fill(dummyVec3s.data(),  0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler())); // filling isn't really necessary, is it?
			mRtAnimObjPositionsBuffer     [i]->fill(dummyVec3s.data(),  0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler())); // filling isn't really necessary, is it?
			mRtAnimObjNTBOffsetBuffer     [i]->fill(dummyNTBOff.data(), 0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler())); // filling isn't really necessary, is it?
			mRtAnimObjNormalsBuffer       [i].enable_shared_ownership();
			mRtAnimObjTangentsBuffer      [i].enable_shared_ownership();
			mRtAnimObjBitangentsBuffer    [i].enable_shared_ownership();
			mRtAnimObjPositionsBuffer     [i].enable_shared_ownership();
			mRtAnimObjNormalsBufferView   [i] = context().create_buffer_view(shared(mRtAnimObjNormalsBuffer   [i]));
			mRtAnimObjTangentsBufferView  [i] = context().create_buffer_view(shared(mRtAnimObjTangentsBuffer  [i]));
			mRtAnimObjBitangentsBufferView[i] = context().create_buffer_view(shared(mRtAnimObjBitangentsBuffer[i]));
			mRtAnimObjPositionsBufferView [i] = context().create_buffer_view(shared(mRtAnimObjPositionsBuffer [i]));
			// note: storage buffers with vec3's are bad! that's why we use a uniform_texel_buffer for normals
		}


		// create ray tracing pipeline
		mPipelineRayTrace = context().create_ray_tracing_pipeline_for(
			define_shader_table(
				ray_generation_shader("shaders/rt_test.rgen.spv"),
				triangles_hit_group::create_with_rahit_and_rchit("shaders/rt_test_transp.rahit.spv", "shaders/rt_test.rchit.spv"),
				triangles_hit_group::create_with_rahit_and_rchit("shaders/rt_test_transp.rahit.spv", "shaders/rt_test_shadowray.rchit.spv"),
				miss_shader("shaders/rt_test.rmiss.spv"),
				miss_shader("shaders/rt_test_shadowray.rmiss.spv")
			),
			context().get_max_ray_tracing_recursion_depth(),
			push_constant_binding_data{shader_type::ray_generation | shader_type::closest_hit, 0, sizeof(push_constant_data_for_rt)},
			RAYTRACING_DESCRIPTOR_BINDINGS(0),
			descriptor_binding(3, 0, mRtImageViews[0]->as_storage_image()),	// when rendering, TASS result image is used
			descriptor_binding(3, 1, mRtDummySegMask->as_storage_image())	// when rendering, TASS segmask is used
		);
		mPipelineRayTrace->print_shader_binding_table_groups();

	}

	void update_top_level_acceleration_structure() {
		// update acceleration structure because:
		// - dynamic object has been toggled on/off
		// - dynamic object has been switched to a different model
		// - dynamic object has moved
		// - TODO: animation
		// - or because we need extend a previous update for the remaining frames in flight

		using namespace avk;
		using namespace gvk;

		static int64_t updateUntilFrame  = -1;
		static int64_t rebuildUntilFrame = -1;

		// check if we need to rebuild due to object switch or enable toggle
		bool needRebuild = false;
		for (int iDynObj = 0; iDynObj < mDynObjects.size(); ++iDynObj) {
			auto &dynObj = mDynObjects[iDynObj];
			if ((mMovingObject.enabled && mMovingObject.moverId == iDynObj) != dynObj.mIsInAccelerationStructure) {
				needRebuild = true;
				break;
			}
		}
		// check if we need to update due to object movement
		bool needUpdate = (mMovingObject.enabled && mDynObjects[mMovingObject.moverId].mMovementMatrix_in_AS != mDynObjects[mMovingObject.moverId].mMovementMatrix_current);

		if (needRebuild) rebuildUntilFrame = context().main_window()->current_frame() + context().main_window()->number_of_frames_in_flight() - 1;
		if (needUpdate)  updateUntilFrame  = context().main_window()->current_frame() + context().main_window()->number_of_frames_in_flight() - 1;
		if (context().main_window()->current_frame() > updateUntilFrame && context().main_window()->current_frame() > rebuildUntilFrame) return;

		auto fif = context().main_window()->in_flight_index_for_frame();

		// clear all mIsInAccelerationStructure flags
		for (auto &dynObj : mDynObjects) dynObj.mIsInAccelerationStructure = false;

		mAllGeometryInstances = mSceneData.mGeometryInstances;

		if (mMovingObject.enabled) {
			// add geometry instances for active object
			auto &dynObj = mDynObjects[mMovingObject.moverId];
			dynObj.mIsInAccelerationStructure = true;
			dynObj.mMovementMatrix_in_AS = dynObj.mMovementMatrix_current;

			for (auto &part : dynObj.mParts) {
				auto tform = dynObj.mIsAnimated ? dynObj.mBaseTransform : dynObj.mBaseTransform * part.mMeshTransform; // don't use the mesh transform for animated models, this is taken care of by bone animation
				auto &mesh = dynObj.mMeshData[part.mMeshIndex];
				auto geoInstance = context().create_geometry_instance(mesh.mBLAS);
				geoInstance.set_transform_column_major(to_array(dynObj.mMovementMatrix_in_AS * tform));
				geoInstance.set_custom_index(dynObj.mRtCustomIndexBase + part.mMeshIndex);
				geoInstance.set_mask(RAYTRACING_CULLMASK_OPAQUE);
				mAllGeometryInstances.push_back(std::move(geoInstance));
			}

		}

		// rebuild or update TLAS for the current frame in flight
		if (context().main_window()->current_frame() <= rebuildUntilFrame) {
			// full rebuild
			mSceneData.mTLASs[fif]->build(mAllGeometryInstances, {}, avk::sync::with_barriers(
				gvk::context().main_window()->command_buffer_lifetime_handler(),
				// Sync before building the TLAS:
				[](avk::command_buffer_t& commandBuffer, avk::pipeline_stage destinationStage, std::optional<avk::read_memory_access> readAccess){
					assert(avk::pipeline_stage::acceleration_structure_build == destinationStage);
					assert(!readAccess.has_value() || avk::memory_access::acceleration_structure_read_access == readAccess.value().value());
					// Wait on all the BLAS builds that happened before (and their memory):
					commandBuffer.establish_global_memory_barrier_rw(
						avk::pipeline_stage::acceleration_structure_build, destinationStage,
						avk::memory_access::acceleration_structure_write_access, readAccess
					);
				},
				// Whatever comes after must wait for this TLAS build to finish (and its memory to be made available):
				//   However, that's exactly what the default_handler_after_operation
				//   does, so let's just use that (it is also the default value for
				//   this handler)
				avk::sync::presets::default_handler_after_operation
			));
			PRINT_DEBUGMARK("TLAS-Rebuild");
		} else {
			// just update existing TLAS
			mSceneData.mTLASs[fif]->update(mAllGeometryInstances, {}, avk::sync::with_barriers(
				gvk::context().main_window()->command_buffer_lifetime_handler(),
				{}, // Nothing to wait for
				// NOTE: this causes a validation error now - bug in current vulkan sdk (1.2.170.0)? -> see todo.txt
				// THEREFORE TEMPORARILY REPLACED BY WAIT IDLE - FIXME!!!
				//[](avk::command_buffer_t& commandBuffer, avk::pipeline_stage srcStage, std::optional<avk::write_memory_access> srcAccess){
				//	// We want this update to be as efficient/as tight as possible
				//	commandBuffer.establish_global_memory_barrier_rw(
				//		srcStage, avk::pipeline_stage::ray_tracing_shaders, // => ray tracing shaders must wait on the building of the acceleration structure
				//		srcAccess, avk::memory_access::acceleration_structure_read_access // TLAS-update's memory must be made visible to ray tracing shader's caches (so they can read from)
				//	);
				//}
				{} // <- FIXME
			));
			context().device().waitIdle(); // <- FIXME
			PRINT_DEBUGMARK("TLAS-Update");
		}
	}

	void update_bottom_level_acceleration_structures() {
		// this updates the BLASs of animated objects
		// TODO: update over multiple fifs, like above?

		using namespace avk;
		using namespace gvk;

		if (!mMovingObject.enabled) return;
		auto &dynObj = mDynObjects[mMovingObject.moverId];
		if (!dynObj.mIsAnimated) return;

		vk::BufferUsageFlags bufferUsage = { vk::BufferUsageFlagBits::eShaderDeviceAddressKHR };

		auto fif = context().main_window()->in_flight_index_for_frame();

		const bool alsoUpdateTLAS = true;
		// seems like updating the BLASs is not enough, if the changed positions leave the initial AABB -> looks cut off (e.g. dragon, dude)
		// so update the TLASs afterwards too, then it's fine

		// blas index buffer stays the same, but the vertex buffer needs to be changed   TODO: -> compute shader
		std::vector<glm::vec3> allNewNormals, allNewTangents, allNewBitangents, allNewPositions;
		for (int iMesh = 0; iMesh < dynObj.mMeshData.size(); ++iMesh) {
			auto &mesh = dynObj.mMeshData[iMesh];

			//auto newPositions = calc_new_mesh_positions_for_testing_only(mesh.mPositions);
			auto [newPositions, newNormals, newTangents, newBitangents] = calc_new_mesh_positions_and_ntb_for_animated_object(dynObj, iMesh);

			auto tmpIndexBuffer		= context().create_buffer(memory_usage::device, bufferUsage, index_buffer_meta::create_from_data(mesh.mIndices));
			auto tmpPositionsBuffer	= context().create_buffer(memory_usage::device, bufferUsage, vertex_buffer_meta::create_from_data(newPositions).describe_only_member(newPositions[0], content_description::position));
			tmpIndexBuffer.enable_shared_ownership();
			tmpPositionsBuffer.enable_shared_ownership();
			tmpIndexBuffer		->fill(mesh.mIndices.data(),   0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler())); // FIXME - sync ok?
			tmpPositionsBuffer	->fill(newPositions.data(),    0, sync::with_barriers(context().main_window()->command_buffer_lifetime_handler())); // FIXME - sync ok?

			mesh.mBLAS->update({ vertex_index_buffer_pair{tmpPositionsBuffer, tmpIndexBuffer} }, {},
				avk::sync::with_barriers(
					[posBfr = tmpPositionsBuffer, idxBfr = tmpIndexBuffer](avk::command_buffer cb) {
						cb->set_custom_deleter([lPosBfr = std::move(posBfr), lIdxBfr = std::move(idxBfr)](){});
						gvk::context().main_window()->handle_lifetime(avk::owned(cb));
					},
					{}, {}
				)
			);

			insert_into(allNewNormals,    newNormals);
			insert_into(allNewTangents,   newTangents);
			insert_into(allNewBitangents, newBitangents);
			insert_into(allNewPositions,  newPositions);
		}

		if (alsoUpdateTLAS) {
			// update TLAS
			// !be sure that at this point the TLAS is built including the current dynObj! -> otherwise may lose device
			//PRINT_DEBUGMARK("before anim TLAS-Update");
			mSceneData.mTLASs[fif]->update(mAllGeometryInstances, {}, avk::sync::with_barriers(
				gvk::context().main_window()->command_buffer_lifetime_handler(),
				// Sync before building the TLAS:
				[](avk::command_buffer_t& commandBuffer, avk::pipeline_stage destinationStage, std::optional<avk::read_memory_access> readAccess) {
					assert(avk::pipeline_stage::acceleration_structure_build == destinationStage);
					assert(!readAccess.has_value() || avk::memory_access::acceleration_structure_read_access == readAccess.value().value());
					// Wait on all the BLAS builds that happened before (and their memory):
					commandBuffer.establish_global_memory_barrier_rw(
						avk::pipeline_stage::acceleration_structure_build, destinationStage,
						avk::memory_access::acceleration_structure_write_access, readAccess
					);
				},
				// NOTE: this causes a validation error now - bug in current vulkan sdk (1.2.170.0)? -> see todo.txt
				// THEREFORE TEMPORARILY REPLACED BY WAIT IDLE - FIXME!!!
				//[](avk::command_buffer_t& commandBuffer, avk::pipeline_stage srcStage, std::optional<avk::write_memory_access> srcAccess) {
				//	// We want this update to be as efficient/as tight as possible
				//	commandBuffer.establish_global_memory_barrier_rw(
				//		srcStage, avk::pipeline_stage::ray_tracing_shaders, // => ray tracing shaders must wait on the building of the acceleration structure
				//		srcAccess, avk::memory_access::acceleration_structure_read_access // TLAS-update's memory must be made visible to ray tracing shader's caches (so they can read from)
				//	);
				//}
				{} // <- FIXME
			));
			context().device().waitIdle(); // <- FIXME
			//PRINT_DEBUGMARK("after anim TLAS-Update");
		} else {
			context().device().waitIdle(); // FIXME - wait for all BLASs to be rebuilt
		}

		// update normals- and vertex-count buffers for anim object
		update_ntb_buffers_for_animated_object(dynObj, allNewNormals, allNewTangents, allNewBitangents, allNewPositions, fif);
	}

	std::tuple<std::vector<glm::vec3>,std::vector<glm::vec3>,std::vector<glm::vec3>,std::vector<glm::vec3>> calc_new_mesh_positions_and_ntb_for_animated_object(dynamic_object &dynObj, int iMeshIndex) {
		// returns tuple { positions, normals, tangents, bitangents }
		// bone matrices have to be calculated already
		// TODO: do this in a compute shader
		auto &mesh = dynObj.mMeshData[iMeshIndex];
		int bonesBaseIndex = iMeshIndex * MAX_BONES;
		auto &posOrg = dynObj.mMeshData[iMeshIndex].mPositions;
		auto &nrmOrg = dynObj.mMeshData[iMeshIndex].mNormals;
		auto &tanOrg = dynObj.mMeshData[iMeshIndex].mTangents;
		auto &bitOrg = dynObj.mMeshData[iMeshIndex].mBitangents;

		const size_t numVertices = posOrg.size();
		assert(nrmOrg.size() == numVertices);

		std::vector<glm::vec3> posNew(numVertices), nrmNew(numVertices), tanNew(numVertices), bitNew(numVertices);
		
		for (auto i = 0; i < numVertices; ++i) {
			auto &aBoneIndices = mesh.mBoneIndices[i];
			auto &aBoneWeights = mesh.mBoneWeights[i];
			glm::mat4 boneMat =
				dynObj.mBoneMatrices[bonesBaseIndex + aBoneIndices[0]] * aBoneWeights[0] +
				dynObj.mBoneMatrices[bonesBaseIndex + aBoneIndices[1]] * aBoneWeights[1] +
				dynObj.mBoneMatrices[bonesBaseIndex + aBoneIndices[2]] * aBoneWeights[2] +
				dynObj.mBoneMatrices[bonesBaseIndex + aBoneIndices[3]] * aBoneWeights[3];

			posNew[i] = glm::vec3(boneMat * glm::vec4(posOrg[i], 1.0f));	// TODO: afair there was a problem with storing pos in a vec3 in the original raster code? (.w != 1) dragon?

			//glm::mat3 nrmMatrix = glm::mat3(glm::inverse(glm::transpose(boneMat)));
			//nrmNew[i] = glm::normalize(nrmMatrix * glm::normalize(nrmOrg[i]));

			glm::mat4 nrmMatrix = glm::inverse(glm::transpose(boneMat));
			nrmNew[i] = glm::normalize(glm::vec3(nrmMatrix * glm::vec4(glm::normalize(nrmOrg[i]), 0)));
			tanNew[i] = glm::normalize(glm::vec3(nrmMatrix * glm::vec4(glm::normalize(tanOrg[i]), 0)));
			bitNew[i] = glm::normalize(glm::vec3(nrmMatrix * glm::vec4(glm::normalize(bitOrg[i]), 0)));
		}

		return std::make_tuple(std::move(posNew), std::move(nrmNew), std::move(tanNew), std::move(bitNew));
	}

	std::vector<glm::vec3> calc_new_mesh_positions_for_testing_only(std::vector<glm::vec3> &posOrg) {
		std::vector<glm::vec3> posNew = posOrg;

		const float angle_per_sec = glm::radians(180.0f);
		static float angle = 0;
		static double tLast = 0.0;
		double t = gvk::context().get_time();
		if (tLast == 0.0) tLast = t;
		double dt = t - tLast;
		tLast = t;
		angle += angle_per_sec * float(dt);

		float dy = sin(angle);

		for (auto &p : posNew) {
			p.y += dy;
		}

		return posNew;
	}

	void update_ntb_buffers_for_animated_object(dynamic_object &dynObj, std::vector<glm::vec3> &normals, std::vector<glm::vec3> &tangents, std::vector<glm::vec3> &bitangents, std::vector<glm::vec3> &positions, gvk::window::frame_id_t fif) {
		// normals/tangents/bitangents-offset per mesh
		uint32_t cnt = 0;
		std::vector<uint32_t> ntbOff;
		for (auto &mesh : dynObj.mMeshData) {
			ntbOff.push_back(cnt);
			cnt += static_cast<uint32_t>(mesh.mPositions.size());
		}

		// sanity check		
		if (normals.size() != cnt || tangents.size() != cnt || bitangents.size() != cnt || positions.size() != cnt) {
			throw avk::runtime_error("update_normals_buffer_for_animated_object sanity check failed !");
		}

		mRtAnimObjNormalsBuffer   [fif]->fill(normals.data(),    0, 0, cnt           * sizeof(glm::vec3), avk::sync::with_barriers(gvk::context().main_window()->command_buffer_lifetime_handler()));
		mRtAnimObjTangentsBuffer  [fif]->fill(tangents.data(),   0, 0, cnt           * sizeof(glm::vec3), avk::sync::with_barriers(gvk::context().main_window()->command_buffer_lifetime_handler()));
		mRtAnimObjBitangentsBuffer[fif]->fill(bitangents.data(), 0, 0, cnt           * sizeof(glm::vec3), avk::sync::with_barriers(gvk::context().main_window()->command_buffer_lifetime_handler()));
		mRtAnimObjPositionsBuffer [fif]->fill(positions.data(),  0, 0, cnt           * sizeof(glm::vec3), avk::sync::with_barriers(gvk::context().main_window()->command_buffer_lifetime_handler()));
		mRtAnimObjNTBOffsetBuffer [fif]->fill(ntbOff.data(),     0, 0, ntbOff.size() * sizeof(uint32_t),  avk::sync::with_barriers(gvk::context().main_window()->command_buffer_lifetime_handler()));
	}

	void update_acceleration_structures_test() {
		// just a test for only updating the static scene geometries

		using namespace avk;
		using namespace gvk;

		const float angle_per_sec = glm::radians(10.0f);
		static float angle = 0;
		static double tLast = 0.0;
		double t = gvk::context().get_time();
		if (tLast == 0.0) { tLast = t; return; }
		double dt = t - tLast;
		tLast = t;
		angle += angle_per_sec * float(dt);

		auto rotMat = glm::rotate(angle, glm::vec3(0, 1, 0));
		for (auto i = 0; i < mSceneData.mGeometryInstances.size(); ++i) {
			auto &geo = mSceneData.mGeometryInstances[i];
			glm::mat4 m = mSceneData.mDebugGeoInstTransforms[i];
			geo.set_transform_column_major(to_array(rotMat * m));
		}

		auto fif = context().main_window()->in_flight_index_for_frame();
		mSceneData.mTLASs[fif]->update(mSceneData.mGeometryInstances, {}, avk::sync::with_barriers(
			gvk::context().main_window()->command_buffer_lifetime_handler(),
			{}, // Nothing to wait for
			[](avk::command_buffer_t& commandBuffer, avk::pipeline_stage srcStage, std::optional<avk::write_memory_access> srcAccess){
				// We want this update to be as efficient/as tight as possible
				commandBuffer.establish_global_memory_barrier_rw(
					srcStage, avk::pipeline_stage::ray_tracing_shaders, // => ray tracing shaders must wait on the building of the acceleration structure
					srcAccess, avk::memory_access::acceleration_structure_read_access // TLAS-update's memory must be made visible to ray tracing shader's caches (so they can read from)
				);
			}
		));

		PRINT_DEBUGMARK("TLAS-Update-Test");
	}

#endif

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

		// build static scene buffers, and upload them
		std::vector<MeshgroupPerInstanceData> attributesData;
		std::vector<CullingBoundingBox> cullingBbData;
		std::vector<MeshgroupBasicInfoGpu> meshgroupInfoData;
		for (auto i = 0; i < mSceneData.mMeshgroups.size(); ++i) {
			auto &mg = mSceneData.mMeshgroups[i];
			gvk::insert_into(attributesData, mg.perInstanceData);

			// bounding boxes (for GPU frustum culling)
			for (auto &insDat : mg.perInstanceData) {
				glm::vec4 p[8];
				mg.boundingBox_untransformed.getTransformedPointsV4(insDat.modelMatrix, p);
				BoundingBox bb;
				bb.calcFromPoints(8, p);
				cullingBbData.push_back({ glm::vec4(bb.min, 1), glm::vec4(bb.max, 1) });
			}

			MeshgroupBasicInfoGpu mgInf;
			mgInf.materialIndex = mg.materialIndex;
			mgInf.numInstances  = static_cast<uint32_t>(mg.perInstanceData.size());
			mgInf.numIndices    = mg.numIndices;
			mgInf.baseIndex     = mg.baseIndex;
			mgInf.transparent   = mg.hasTransparency;
			meshgroupInfoData.push_back(mgInf);
		}
		assert(mSceneData.mAttributesBuffer->meta<avk::storage_buffer_meta>().total_size() == attributesData.size() * sizeof(MeshgroupPerInstanceData));
		mSceneData.mAttributesBuffer        ->fill(attributesData.data(),    0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
		mSceneData.mCullingBoundingBoxBuffer->fill(cullingBbData.data(),     0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
		mSceneData.mMeshgroupInfoBuffer     ->fill(meshgroupInfoData.data(), 0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));

		// build dynamic scene and drawcommand buffers, and upload them
		rebuild_scene_buffers(0, gvk::context().main_window()->number_of_frames_in_flight() - 1);

		// upload data for other models
		for (auto& dynObj : mDynObjects) {
			for (auto &md : dynObj.mMeshData) {
				md.mIndexBuffer				->fill(md.mIndices.data(),     0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
				md.mPositionsBuffer			->fill(md.mPositions.data(),   0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
				md.mTexCoordsBuffer			->fill(md.mTexCoords.data(),   0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
				md.mNormalsBuffer			->fill(md.mNormals.data(),     0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
				md.mTangentsBuffer			->fill(md.mTangents.data(),    0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
				md.mBitangentsBuffer		->fill(md.mBitangents.data(),  0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
				if (dynObj.mIsAnimated) {
					md.mBoneWeightsBuffer	->fill(md.mBoneWeights.data(), 0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
					md.mBoneIndicesBuffer	->fill(md.mBoneIndices.data(), 0, avk::sync::with_barriers([this](avk::command_buffer cb){ mStoredCommandBuffers.emplace_back(std::move(cb)); }, {}, {}));
				}
			}
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
			vertex_shader("shaders/transform_and_pass_on.vert.spv"),
			fragment_shader("shaders/blinnphong_and_normal_mapping.frag.spv"),
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
			SCENE_DRAW_DESCRIPTOR_BINDINGS(0)
			SHADOWMAP_DESCRIPTOR_BINDINGS(0)
			descriptor_binding(1, 0, mMatricesUserInputBuffer[0]),
			descriptor_binding(1, 1, mLightsourcesBuffer[0])
		);

		mPipelineLightingPass = context().create_graphics_pipeline_for(
			vertex_shader("shaders/lighting_pass.vert.spv"),
			fragment_shader("shaders/lighting_pass.frag.spv"),
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
			SCENE_DRAW_DESCRIPTOR_BINDINGS(0)
			SHADOWMAP_DESCRIPTOR_BINDINGS(0)
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

		const char * vert_shader_name = "shaders/transform_and_pass_on.vert.spv";
		const char * frag_shader_name = "shaders/fwd_geometry.frag.spv";

		mPipelineFwdOpaque = context().create_graphics_pipeline_for(
			vertex_shader(vert_shader_name),
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
			mRenderpass, 0u, // subpass #0
			push_constant_binding_data { shader_type::all, 0, sizeof(push_constant_data_for_dii) }, // We also have to declare that we're going to submit push constants
			descriptor_binding(0, 0, mMaterialBuffer),						// As far as used resources are concerned, we need the materials buffer (type: vk::DescriptorType::eStorageBuffer),
			descriptor_binding(0, 1, mImageSamplers),						// multiple images along with their sampler (array of vk::DescriptorType::eCombinedImageSampler),
			SCENE_DRAW_DESCRIPTOR_BINDINGS(0)
			SHADOWMAP_DESCRIPTOR_BINDINGS(0)
			descriptor_binding(1, 0, mMatricesUserInputBuffer[0]),
			descriptor_binding(1, 1, mLightsourcesBuffer[0])
		);

		// this is almost the same, except for the specialization constant, alpha blending (and backface culling? depth write?)
		mPipelineFwdTransparent = context().create_graphics_pipeline_for(
			vertex_shader(vert_shader_name),
			fragment_shader(frag_shader_name).set_specialization_constant(SPECCONST_ID_TRANSPARENCY, uint32_t{ SPECCONST_VAL_TRANSPARENT }), // transparent pass

			//cfg::color_blending_config::enable_alpha_blending_for_all_attachments(),
			cfg::color_blending_config::enable_alpha_blending_for_attachment(0),
			cfg::culling_mode::disabled,
			// cfg::depth_write::disabled(), // would need back-to-front sorting, also a problem for TAA... so leave it on (and render only stuff with alpha >= threshold)
			// cfg::depth_test::disabled(),  // not good, definitely needs sorting

			from_buffer_binding(0) -> stream_per_vertex<glm::vec3>() -> to_location(0),		// <-- corresponds to vertex shader's aPosition
			from_buffer_binding(1) -> stream_per_vertex<glm::vec2>() -> to_location(1),		// <-- corresponds to vertex shader's aTexCoords
			from_buffer_binding(2) -> stream_per_vertex<glm::vec3>() -> to_location(2),		// <-- corresponds to vertex shader's aNormal
			from_buffer_binding(3) -> stream_per_vertex<glm::vec3>() -> to_location(3),		// <-- corresponds to vertex shader's aTangent
			from_buffer_binding(4) -> stream_per_vertex<glm::vec3>() -> to_location(4),		// <-- corresponds to vertex shader's aBitangent
			// Some further settings:
			cfg::front_face::define_front_faces_to_be_clockwise(),
			cfg::viewport_depth_scissors_config::from_framebuffer(mFramebuffer[0]),
			mRenderpass, 0u, // subpass #0
			push_constant_binding_data { shader_type::all, 0, sizeof(push_constant_data_for_dii) }, // We also have to declare that we're going to submit push constants
			descriptor_binding(0, 0, mMaterialBuffer),						// As far as used resources are concerned, we need the materials buffer (type: vk::DescriptorType::eStorageBuffer),
			descriptor_binding(0, 1, mImageSamplers),						// multiple images along with their sampler (array of vk::DescriptorType::eCombinedImageSampler),
			SCENE_DRAW_DESCRIPTOR_BINDINGS(0)
			SHADOWMAP_DESCRIPTOR_BINDINGS(0)
			descriptor_binding(1, 0, mMatricesUserInputBuffer[0]),
			descriptor_binding(1, 1, mLightsourcesBuffer[0])
		);

		// alternative pipeline - render transparency with simple alpha test only, no blending
		mPipelineFwdTransparentNoBlend = context().create_graphics_pipeline_for(
			vertex_shader(vert_shader_name),
			fragment_shader(frag_shader_name).set_specialization_constant(SPECCONST_ID_TRANSPARENCY, uint32_t{ SPECCONST_VAL_TRANSPARENT }), // transparent pass

			cfg::culling_mode::disabled,
			// cfg::depth_write::disabled(), // would need back-to-front sorting, also a problem for TAA... so leave it on (and render only stuff with alpha >= threshold)
			// cfg::depth_test::disabled(),  // not good, definitely needs sorting

			from_buffer_binding(0) -> stream_per_vertex<glm::vec3>() -> to_location(0),		// <-- corresponds to vertex shader's aPosition
			from_buffer_binding(1) -> stream_per_vertex<glm::vec2>() -> to_location(1),		// <-- corresponds to vertex shader's aTexCoords
			from_buffer_binding(2) -> stream_per_vertex<glm::vec3>() -> to_location(2),		// <-- corresponds to vertex shader's aNormal
			from_buffer_binding(3) -> stream_per_vertex<glm::vec3>() -> to_location(3),		// <-- corresponds to vertex shader's aTangent
			from_buffer_binding(4) -> stream_per_vertex<glm::vec3>() -> to_location(4),		// <-- corresponds to vertex shader's aBitangent
																							// Some further settings:
			cfg::front_face::define_front_faces_to_be_clockwise(),
			cfg::viewport_depth_scissors_config::from_framebuffer(mFramebuffer[0]),
			mRenderpass, 0u, // subpass #0
			push_constant_binding_data { shader_type::all, 0, sizeof(push_constant_data_for_dii) }, // We also have to declare that we're going to submit push constants
			descriptor_binding(0, 0, mMaterialBuffer),						// As far as used resources are concerned, we need the materials buffer (type: vk::DescriptorType::eStorageBuffer),
			descriptor_binding(0, 1, mImageSamplers),						// multiple images along with their sampler (array of vk::DescriptorType::eCombinedImageSampler),
			SCENE_DRAW_DESCRIPTOR_BINDINGS(0)
			SHADOWMAP_DESCRIPTOR_BINDINGS(0)
			descriptor_binding(1, 0, mMatricesUserInputBuffer[0]),
			descriptor_binding(1, 1, mLightsourcesBuffer[0])
		);

	}

	void prepare_common_pipelines() {
		using namespace avk;
		using namespace gvk;

		mPipelineAnimObject = context().create_graphics_pipeline_for(
			vertex_shader("shaders/animatedObject.vert.spv"),
#if FORWARD_RENDERING
			fragment_shader("shaders/fwd_geometry.frag.spv").set_specialization_constant(SPECCONST_ID_TRANSPARENCY, uint32_t{ SPECCONST_VAL_OPAQUE }), //  opaque pass
#else
			fragment_shader("shaders/blinnphong_and_normal_mapping.frag.spv"),
#endif
			from_buffer_binding(0) -> stream_per_vertex<glm::vec3>() -> to_location(0),		// aPosition
			from_buffer_binding(1) -> stream_per_vertex<glm::vec2>() -> to_location(1),		// aTexCoords
			from_buffer_binding(2) -> stream_per_vertex<glm::vec3>() -> to_location(2),		// aNormal
			from_buffer_binding(3) -> stream_per_vertex<glm::vec3>() -> to_location(3),		// aTangent
			from_buffer_binding(4) -> stream_per_vertex<glm::vec3>() -> to_location(4),		// aBitangent
			from_buffer_binding(5) -> stream_per_vertex<glm::vec4>() -> to_location(5),		// aBoneWeights
			from_buffer_binding(6) -> stream_per_vertex<glm::uvec4>()-> to_location(6),		// aBoneIndices
			cfg::front_face::define_front_faces_to_be_counter_clockwise(),
			cfg::viewport_depth_scissors_config::from_framebuffer(mFramebuffer[0]),
			mRenderpass, 0u, // subpass #0
			push_constant_binding_data { shader_type::all, 0, sizeof(push_constant_data_for_dii) },
			descriptor_binding(0, 0, mMaterialBuffer),
			descriptor_binding(0, 1, mImageSamplers),
			SCENE_DRAW_DESCRIPTOR_BINDINGS(0)
			SHADOWMAP_DESCRIPTOR_BINDINGS(0)
			descriptor_binding(1, 0, mMatricesUserInputBuffer[0]),
			descriptor_binding(1, 1, mLightsourcesBuffer[0]),
			descriptor_binding(3, 0, mBoneMatricesBuffer[0]),
			descriptor_binding(3, 1, mBoneMatricesPrevBuffer[0])
		);

		uint32_t subpass = (FORWARD_RENDERING) ? 1u : 2u;

		mPipelineDrawCamPath = context().create_graphics_pipeline_for(
			vertex_shader("shaders/drawpath.vert.spv"),
			fragment_shader("shaders/drawpath.frag.spv"),
			from_buffer_binding(0) -> stream_per_vertex<glm::vec4>() -> to_location(0),		// <-- aPositionAndId
			cfg::primitive_topology::points,
			cfg::depth_write::disabled(),
			cfg::viewport_depth_scissors_config::from_framebuffer(mFramebuffer[0]),
			mRenderpass, subpass,
			descriptor_binding(1, 0, mMatricesUserInputBuffer[0])
		);

		mPipelineTestImage = context().create_graphics_pipeline_for(
			vertex_shader("shaders/testimage.vert.spv"),
			fragment_shader("shaders/testimage.frag.spv"),
			from_buffer_binding(0)->stream_per_vertex(&helpers::quad_vertex::mPosition)->to_location(0),
			from_buffer_binding(0)->stream_per_vertex(&helpers::quad_vertex::mTextureCoordinate)->to_location(1),
			cfg::front_face::define_front_faces_to_be_clockwise(),
			cfg::culling_mode::disabled,
			cfg::depth_test::disabled(),
			cfg::viewport_depth_scissors_config::from_framebuffer(mFramebuffer[0]),
			mRenderpass, subpass,
			descriptor_binding(0, 0, mTestImageSampler_bilinear),
			descriptor_binding(0, 1, mTestImages[0]),
			descriptor_binding(1, 0, mMatricesUserInputBuffer[0])
		);

#if ENABLE_SHADOWMAP
		mPipelineShadowmapOpaque = context().create_graphics_pipeline_for(
			vertex_shader("shaders/shadowmap.vert.spv"), // no fragment shader
			from_buffer_binding(0) -> stream_per_vertex<glm::vec3>() -> to_location(0),
			mShadowmapRenderpass,
			cfg::culling_mode::disabled,	// no backface culling // (for now) TODO - enabling it leads to shadow problems!
			cfg::front_face::define_front_faces_to_be_counter_clockwise(),
			cfg::viewport_depth_scissors_config::from_framebuffer(mShadowmapPerCascade[0].mShadowmapFramebuffer[0]),
			cfg::depth_clamp_bias::dynamic(), // allow dynamic setting of depth bias AND enable it
			push_constant_binding_data { shader_type::all, 0, sizeof(push_constant_data_for_dii) },
			descriptor_binding(0, 0, mMaterialBuffer),
			descriptor_binding(0, 1, mImageSamplers),						// need to sample alpha (not here, but maintain layout compatibility)
			SCENE_DRAW_DESCRIPTOR_BINDINGS(0)
			descriptor_binding(1, 0, mMatricesUserInputBuffer[0]),
			descriptor_binding(1, 1, mLightsourcesBuffer[0])				// compat
		);

		mPipelineShadowmapTransparent = context().create_graphics_pipeline_for(
			vertex_shader("shaders/shadowmap_transparent.vert.spv"),
			fragment_shader("shaders/shadowmap_transparent.frag.spv"),
			from_buffer_binding(0) -> stream_per_vertex<glm::vec3>() -> to_location(0),
			from_buffer_binding(1) -> stream_per_vertex<glm::vec2>() -> to_location(1),		// aTexCoords
			mShadowmapRenderpass,
			cfg::culling_mode::disabled,	// no backface culling // (for now) TODO
			cfg::viewport_depth_scissors_config::from_framebuffer(mShadowmapPerCascade[0].mShadowmapFramebuffer[0]),
			cfg::depth_clamp_bias::dynamic(), // allow dynamic setting of depth bias AND enable it
			push_constant_binding_data { shader_type::all, 0, sizeof(push_constant_data_for_dii) },
			descriptor_binding(0, 0, mMaterialBuffer),
			descriptor_binding(0, 1, mImageSamplers),						// need to sample alpha
			SCENE_DRAW_DESCRIPTOR_BINDINGS(0)
			descriptor_binding(1, 0, mMatricesUserInputBuffer[0]),
			descriptor_binding(1, 1, mLightsourcesBuffer[0])				// compat
		);

		mPipelineShadowmapAnimObject = context().create_graphics_pipeline_for(
			vertex_shader("shaders/shadowmap_animated.vert.spv"),	// no frag shader
			from_buffer_binding(0) -> stream_per_vertex<glm::vec3>() -> to_location(0),		// aPosition
			from_buffer_binding(1) -> stream_per_vertex<glm::vec4>() -> to_location(1),		// aBoneWeights
			from_buffer_binding(2) -> stream_per_vertex<glm::uvec4>()-> to_location(2),		// aBoneIndices
			mShadowmapRenderpass,
			cfg::front_face::define_front_faces_to_be_counter_clockwise(),
			cfg::viewport_depth_scissors_config::from_framebuffer(mShadowmapPerCascade[0].mShadowmapFramebuffer[0]),
			cfg::depth_clamp_bias::dynamic(), // allow dynamic setting of depth bias AND enable it
			push_constant_binding_data { shader_type::all, 0, sizeof(push_constant_data_for_dii) },
			descriptor_binding(0, 0, mMaterialBuffer),
			descriptor_binding(0, 1, mImageSamplers),
			SCENE_DRAW_DESCRIPTOR_BINDINGS(0)
			descriptor_binding(1, 0, mMatricesUserInputBuffer[0]),
			descriptor_binding(1, 1, mLightsourcesBuffer[0]),				// compat
			descriptor_binding(3, 0, mBoneMatricesBuffer[0]),
			descriptor_binding(3, 1, mBoneMatricesPrevBuffer[0])			// compat
		);

		mPipelineDrawShadowmap = context().create_graphics_pipeline_for(
			vertex_shader("shaders/draw_shadowmap.vert.spv"),
			fragment_shader("shaders/draw_shadowmap.frag.spv"),
			from_buffer_binding(0)->stream_per_vertex(&helpers::quad_vertex::mPosition)->to_location(0),
			from_buffer_binding(0)->stream_per_vertex(&helpers::quad_vertex::mTextureCoordinate)->to_location(1),
			cfg::front_face::define_front_faces_to_be_clockwise(),
			cfg::culling_mode::disabled,
			cfg::depth_test::disabled(),
			cfg::viewport_depth_scissors_config::from_framebuffer(mFramebuffer[0]),
			mRenderpass, subpass,
			descriptor_binding(1, 0, mMatricesUserInputBuffer[0]),
			SHADOWMAP_DESCRIPTOR_BINDINGS(0)
			descriptor_binding(5, 0, mGenericSamplerNearestNeighbour)
		);

		mPipelineDrawFrustum = context().create_graphics_pipeline_for(
			vertex_shader("shaders/draw_frustum.vert.spv"),
			fragment_shader("shaders/draw_frustum.frag.spv"),
			cfg::primitive_topology::lines,
			cfg::culling_mode::disabled,
			//cfg::depth_test::disabled(),
			cfg::viewport_depth_scissors_config::from_framebuffer(mFramebuffer[0]),
			mRenderpass, subpass,
			descriptor_binding(1, 0, mMatricesUserInputBuffer[0])
		);
#endif

		// pipelines for GPU frustum culling (and scene drawcommand building)
		mPipelineFrustumCulling = context().create_compute_pipeline_for(
			compute_shader("shaders/frustum_culling.comp.spv"),
			descriptor_binding(0, 0, mSceneData.mCullingUniformsBuffer[0]),
			descriptor_binding(0, 1, mSceneData.mCullingVisibilityBuffer[0]),
			descriptor_binding(0, 2, mSceneData.mCullingBoundingBoxBuffer)
		);

		mPipelineBuildSceneBuffers = context().create_compute_pipeline_for(
			compute_shader("shaders/build_scene_buffers.comp.spv"),
			descriptor_binding(0, 0, mSceneData.mCullingUniformsBuffer[0]),
			descriptor_binding(0, 1, mSceneData.mCullingVisibilityBuffer[0]),
			descriptor_binding(0, 2, mSceneData.mMeshgroupInfoBuffer),
			descriptor_binding(0, 3, mSceneData.mDrawnMeshgroupBuffer[0]),
			descriptor_binding(0, 4, mSceneData.mDrawnMeshAttribIndexBuffer[0]),
			descriptor_binding(0, 5, mSceneData.mMeshgroupsLayoutInfoBuffer[0]),
			descriptor_binding(0, 6, mSceneData.mDrawCommandsBuffer[0]->as_storage_buffer()),
			descriptor_binding(0, 7, mSceneData.mDrawCountBuffer[0]->as_storage_buffer()),
			push_constant_binding_data{ shader_type::compute, 0, sizeof(BuildSceneBuffersPushConstants) }
		);
	}

	void print_pipeline_info(avk::graphics_pipeline *pPipe, std::string desc) {
		if (!pPipe->has_value()) return;
		auto &pipe = pPipe->get();
		printf("%-30s: handle %016llx layout %016llx\n", desc.c_str(), (uint64_t)(void *)(pipe.handle()), (uint64_t)(void *)(pipe.layout_handle()));
	}

	void print_pipelines_info() {
		print_pipeline_info(&mPipelineFwdOpaque,				"mPipelineFwdOpaque");
		print_pipeline_info(&mPipelineFwdTransparent,			"mPipelineFwdTransparent");
		print_pipeline_info(&mPipelineFwdTransparentNoBlend,	"mPipelineFwdTransparentNoBlend");
		print_pipeline_info(&mPipelineFirstPass,				"mPipelineFirstPass");
		print_pipeline_info(&mPipelineLightingPass,				"mPipelineLightingPass");
		print_pipeline_info(&mPipelineAnimObject,				"mPipelineAnimObject");
		print_pipeline_info(&mPipelineDrawCamPath,				"mPipelineDrawCamPath");
		print_pipeline_info(&mPipelineDrawFrustum,				"mPipelineDrawFrustum");
		print_pipeline_info(&mPipelineDrawShadowmap,			"mPipelineDrawShadowmap");
		print_pipeline_info(&mPipelineShadowmapAnimObject,		"mPipelineShadowmapAnimObject");
		print_pipeline_info(&mPipelineShadowmapOpaque,			"mPipelineShadowmapOpaque");
		print_pipeline_info(&mPipelineShadowmapTransparent,		"mPipelineShadowmapTransparent");
		print_pipeline_info(&mPipelineTestImage,				"mPipelineTestImage");
		print_pipeline_info(&mSkyboxPipeline,					"mSkyboxPipeline");
	}

	void draw_scene(avk::command_buffer &cmd, gvk::window::frame_id_t fif, bool transparentParts, int shadowCascade = -1) {
		using namespace avk;

		// set depth bias in shadow pass
		if (shadowCascade >= 0) {
			// TODO: is there no avk-method to set depth bias?
			if (mShadowMap.depthBias[shadowCascade].enable) {
				cmd->handle().setDepthBias(mShadowMap.depthBias[shadowCascade].constant, mShadowMap.depthBias[shadowCascade].clamp, mShadowMap.depthBias[shadowCascade].slope);
			} else {
				cmd->handle().setDepthBias(0.f, 0.f, 0.f);
			}
		}

		cmd->draw_indexed_indirect_count(
			const_referenced(mSceneData.mDrawCommandsBuffer[fif]),
			const_referenced(mSceneData.mIndexBuffer),
			transparentParts ? mSceneData.mMaxTransparentMeshgroups : mSceneData.mMaxOpaqueMeshgroups,
			vk::DeviceSize{ transparentParts ? mSceneData.mMaxOpaqueMeshgroups * sizeof(vk::DrawIndexedIndirectCommand) : 0 },
			static_cast<uint32_t>(sizeof(vk::DrawIndexedIndirectCommand)),
			const_referenced(mSceneData.mDrawCountBuffer[fif]),
			vk::DeviceSize{ transparentParts ? sizeof(uint32_t) : 0 },
			const_referenced(mSceneData.mPositionsBuffer),
			const_referenced(mSceneData.mTexCoordsBuffer),
			const_referenced(mSceneData.mNormalsBuffer),
			const_referenced(mSceneData.mTangentsBuffer),
			const_referenced(mSceneData.mBitangentsBuffer)
		);
	}

	void draw_dynamic_objects(avk::command_buffer &cmd, gvk::window::frame_id_t fif, int shadowCascade = -1) {
		if (!mMovingObject.enabled) return;

		bool shadowPass = shadowCascade >= 0;

		push_constant_data_for_dii pushc_dii = {};
		if (shadowPass) pushc_dii.mShadowMapCascadeToBuild = shadowCascade;

		if (mDynObjects[mMovingObject.moverId].mIsAnimated) {
			// animated
			rdoc::beginSection(cmd->handle(), "render dynamic object (animated)");
			auto &pipe = shadowPass ? mPipelineShadowmapAnimObject : mPipelineAnimObject;

			cmd->bind_pipeline(const_referenced(pipe));

			if (shadowPass) {
				cmd->bind_descriptors(pipe->layout(), mDescriptorCache.get_or_create_descriptor_sets({
					descriptor_binding(0, 0, mMaterialBuffer),
					descriptor_binding(0, 1, mImageSamplers),
					SCENE_DRAW_DESCRIPTOR_BINDINGS(fif)
					//SHADOWMAP_DESCRIPTOR_BINDINGS(fif)
					descriptor_binding(1, 0, mMatricesUserInputBuffer[fif]),
					descriptor_binding(1, 1, mLightsourcesBuffer[fif]),
					descriptor_binding(3, 0, mBoneMatricesBuffer[fif]),
					descriptor_binding(3, 1, mBoneMatricesPrevBuffer[fif]),
					}));
			} else {
				cmd->bind_descriptors(pipe->layout(), mDescriptorCache.get_or_create_descriptor_sets({
					descriptor_binding(0, 0, mMaterialBuffer),
					descriptor_binding(0, 1, mImageSamplers),
					SCENE_DRAW_DESCRIPTOR_BINDINGS(fif)
					SHADOWMAP_DESCRIPTOR_BINDINGS(fif)
					descriptor_binding(1, 0, mMatricesUserInputBuffer[fif]),
					descriptor_binding(1, 1, mLightsourcesBuffer[fif]),
					descriptor_binding(3, 0, mBoneMatricesBuffer[fif]),
					descriptor_binding(3, 1, mBoneMatricesPrevBuffer[fif]),
					}));
			}

			auto &dynObj = mDynObjects[mMovingObject.moverId];
			for (auto &part : dynObj.mParts) {
				auto &md = dynObj.mMeshData[part.mMeshIndex];

				pushc_dii.mDrawType               = -(mMovingObject.moverId + 1);
				pushc_dii.mMover_baseModelMatrix  = dynObj.mBaseTransform; // no longer multiply with part.mMeshTransform, as bone matrices are already computed in object space now;
				pushc_dii.mMover_materialIndex    = md.mMaterialIndex;
				pushc_dii.mMover_meshIndex        = part.mMeshIndex;

				cmd->push_constants(pipe->layout(), pushc_dii);
				if (shadowPass) {
					cmd->draw_indexed(
						avk::const_referenced(md.mIndexBuffer),
						avk::const_referenced(md.mPositionsBuffer),
						avk::const_referenced(md.mBoneWeightsBuffer),
						avk::const_referenced(md.mBoneIndicesBuffer)
					);
				} else {
					cmd->draw_indexed(
						avk::const_referenced(md.mIndexBuffer),
						avk::const_referenced(md.mPositionsBuffer),
						avk::const_referenced(md.mTexCoordsBuffer),
						avk::const_referenced(md.mNormalsBuffer),
						avk::const_referenced(md.mTangentsBuffer),
						avk::const_referenced(md.mBitangentsBuffer),
						avk::const_referenced(md.mBoneWeightsBuffer),
						avk::const_referenced(md.mBoneIndicesBuffer)
					);
				}
			}
			rdoc::endSection(cmd->handle());
		} else {
			// not animated
			rdoc::beginSection(cmd->handle(), "render dynamic object (not animated)");
			auto &dynObj = mDynObjects[mMovingObject.moverId];
			for (auto &part : dynObj.mParts) {
				auto &md = dynObj.mMeshData[part.mMeshIndex];

				pushc_dii.mDrawType               = -(mMovingObject.moverId + 1);
				pushc_dii.mMover_baseModelMatrix  = dynObj.mBaseTransform * part.mMeshTransform;
				pushc_dii.mMover_materialIndex    = md.mMaterialIndex;
				pushc_dii.mMover_meshIndex        = part.mMeshIndex;

				if (shadowPass) {
					cmd->push_constants(mPipelineShadowmapOpaque->layout(), pushc_dii);
					cmd->draw_indexed(
						avk::const_referenced(md.mIndexBuffer),
						avk::const_referenced(md.mPositionsBuffer)
					);
				} else {
					const auto &pipe = FORWARD_RENDERING ? mPipelineFwdOpaque : mPipelineFirstPass;
					cmd->push_constants(pipe->layout(), pushc_dii);
					cmd->draw_indexed(
						avk::const_referenced(md.mIndexBuffer),
						avk::const_referenced(md.mPositionsBuffer),
						avk::const_referenced(md.mTexCoordsBuffer),
						avk::const_referenced(md.mNormalsBuffer),
						avk::const_referenced(md.mTangentsBuffer),
						avk::const_referenced(md.mBitangentsBuffer)
					);
				}
			}
			rdoc::endSection(cmd->handle());
		}
	}

	void invalidate_command_buffers() {
		mReRecordCommandBuffers = true;
	}

	void alloc_command_buffers_for_models() {
		auto& commandPool = gvk::context().get_command_pool_for_resettable_command_buffers(*mQueue); // resettable: we may need to re-record them or re-use them
		auto fif = gvk::context().main_window()->number_of_frames_in_flight();
		for (decltype(fif) i=0; i < fif; ++i) {
			if (!(mModelsCommandBuffer[i].has_value())) mModelsCommandBuffer[i] = commandPool->alloc_command_buffer();
		}
	}

	void compute_frustum_culling(avk::command_buffer &cmd, gvk::window::frame_id_t fif) {
#if ENABLE_GPU_FRUSTUM_CULLING
		// frustum culling

		if (mSceneData.mRegeneratePerFrame) {
			using namespace avk;
			using namespace gvk;

			rdoc::beginSection(cmd->handle(), "Frustum culling", fif);

			// do frustum culling
			cmd->bind_pipeline(const_referenced(mPipelineFrustumCulling));
			cmd->bind_descriptors(mPipelineFrustumCulling->layout(), mDescriptorCache.get_or_create_descriptor_sets({
				descriptor_binding(0, 0, mSceneData.mCullingUniformsBuffer[fif]),
				descriptor_binding(0, 1, mSceneData.mCullingVisibilityBuffer[fif]),
				descriptor_binding(0, 2, mSceneData.mCullingBoundingBoxBuffer)
				}));
			cmd->handle().dispatch(static_cast<uint32_t>((mSceneData.mNumTotalInstances + GPU_FRUSTUM_CULLING_WORKGROUP_SIZE - 1) / GPU_FRUSTUM_CULLING_WORKGROUP_SIZE), 1u, 1u);

			// the resulting visibility buffer will be read in a compute shader
			cmd->establish_global_memory_barrier(
				pipeline_stage::compute_shader,                        /* -> */ pipeline_stage::compute_shader,
				memory_access::shader_buffers_and_images_write_access, /* -> */ memory_access::shader_buffers_and_images_read_access
			);
			rdoc::endSection(cmd->handle());
		}
#endif
	}

	void compute_scene_draw_buffers(avk::command_buffer &cmd, gvk::window::frame_id_t fif, int shadowCascade = -1) {
#if ENABLE_GPU_FRUSTUM_CULLING
		// build scene draw buffers

		if (mSceneData.mRegeneratePerFrame) {
			using namespace avk;
			using namespace gvk;

			rdoc::beginSection(cmd->handle(), "Build draw commands", fif);

			cmd->bind_pipeline(const_referenced(mPipelineBuildSceneBuffers));
			cmd->bind_descriptors(mPipelineBuildSceneBuffers->layout(), mDescriptorCache.get_or_create_descriptor_sets({
				descriptor_binding(0, 0, mSceneData.mCullingUniformsBuffer[fif]),
				descriptor_binding(0, 1, mSceneData.mCullingVisibilityBuffer[fif]),
				descriptor_binding(0, 2, mSceneData.mMeshgroupInfoBuffer),
				descriptor_binding(0, 3, mSceneData.mDrawnMeshgroupBuffer[fif]),
				descriptor_binding(0, 4, mSceneData.mDrawnMeshAttribIndexBuffer[fif]),
				descriptor_binding(0, 5, mSceneData.mMeshgroupsLayoutInfoBuffer[fif]),
				descriptor_binding(0, 6, mSceneData.mDrawCommandsBuffer[fif]->as_storage_buffer()),
				descriptor_binding(0, 7, mSceneData.mDrawCountBuffer[fif]->as_storage_buffer()),
				}));

			BuildSceneBuffersPushConstants pushc;
			pushc.frustum = shadowCascade + 1;
			cmd->push_constants(mPipelineBuildSceneBuffers->layout(), pushc);
			cmd->handle().dispatch(1u, 1u, 1u);

			// compute shader wrote storage and indirect buffers - these are used by vertex shaders and indirect draw commands
			cmd->establish_global_memory_barrier(
				pipeline_stage::compute_shader,                        /* -> */ pipeline_stage::vertex_shader | pipeline_stage::draw_indirect,
				memory_access::shader_buffers_and_images_write_access, /* -> */ memory_access::shader_buffers_and_images_read_access | memory_access::indirect_command_data_read_access
			);

			rdoc::endSection(cmd->handle());
		}
#endif
	}

	void do_raytrace_test(avk::command_buffer &cmd, gvk::window::frame_id_t fif, bool taaAssist) {
#if ENABLE_RAYTRACING
		using namespace avk;
		using namespace gvk;

		cmd->bind_pipeline(const_referenced(mPipelineRayTrace));
		cmd->bind_descriptors(mPipelineRayTrace->layout(), mDescriptorCache.get_or_create_descriptor_sets({
			RAYTRACING_DESCRIPTOR_BINDINGS(fif),
			descriptor_binding(3, 0, taaAssist ? mAntiAliasing.getRayTraceInputImage()->get().as_storage_image() : mRtImageViews[0]->as_storage_image()),	// when augmenting, TASS result image is used, else this is ignored
			descriptor_binding(3, 1, taaAssist ? mAntiAliasing.getRayTraceSegMask()   ->get().as_storage_image() : mRtDummySegMask ->as_storage_image())	// when augmenting, TASS segmask is used, else this is ignored
			}));
		push_constant_data_for_rt pushc;
		pushc.mCameraTransform			= mQuakeCam.global_transformation_matrix();
		pushc.mCameraViewProjMatrix		= mOriginalProjMat * mQuakeCam.view_matrix();
		pushc.mLightDir					= glm::vec4(mDirLight.dir, 0.f);
		pushc.mDirLightIntensity		= glm::vec4(mDirLight.intensity * mDirLight.boost, 0.f);
		pushc.mAmbientLightIntensity	= glm::vec4(mAmbLight.col       * mAmbLight.boost, 0.f);
		pushc.mNormalMappingStrength	= mNormalMappingStrength;
		pushc.mMaxRayLength				= glm::distance(mSceneData.mBoundingBox.min, mSceneData.mBoundingBox.max); // use scene BB diagonal as max ray length
		pushc.mNumSamples				= mRtSamplesPerPixel;
		pushc.mAnimObjFirstMeshId		= static_cast<int>(mDynObjects[mMovingObject.moverId].mRtCustomIndexBase);
		pushc.mAnimObjNumMeshes			= static_cast<int>((mMovingObject.enabled && mDynObjects[mMovingObject.moverId].mIsAnimated) ? mDynObjects[mMovingObject.moverId].mMeshData.size() : 0);
		pushc.mDoShadows				= (mShadowMap.enable ? 0x01 : 0x00) | (mShadowMap.enableForTransparency ? 0x02 : 0x00);
		pushc.mAugmentTAA				= taaAssist ? VK_TRUE : VK_FALSE;
		pushc.mAugmentTAADebug			= taaAssist && mRtDebugSparse ? VK_TRUE : VK_FALSE;
		pushc.mApproximateLod			= mRtApproximateLod;
		cmd->handle().pushConstants(mPipelineRayTrace->layout_handle(), vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR, 0, sizeof(pushc), &pushc);
		cmd->trace_rays(
			//for_each_pixel(context().main_window()),
			vk::Extent3D{ mLoResolution.x, mLoResolution.y, 1u },
			mPipelineRayTrace->shader_binding_table(),
			using_raygen_group_at_index(0),
			using_miss_group_at_index(0),
			using_hit_group_at_index(0)
		);

#endif
	}

	void record_all_command_buffers_for_models() {
		auto fif = gvk::context().main_window()->number_of_frames_in_flight();
		for (decltype(fif) i=0; i < fif; ++i) {
			record_single_command_buffer_for_models(mModelsCommandBuffer[i], i);
		}
	}

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

		push_constant_data_for_dii pushc_dii = {};

		commandBuffer->begin_recording();

#if ENABLE_RAYTRACING
		if (mDoRayTraceTest) {
			do_raytrace_test(commandBuffer, fif, false);
			commandBuffer->begin_render_pass_for_framebuffer(firstPipe->get_renderpass(), mFramebuffer[fif]);
			commandBuffer->next_subpass();
			commandBuffer->end_render_pass();
			blit_image(mRtImageViews[fif]->get_image(), mFramebuffer[fif]->image_view_at(0)->get_image(), sync::with_barriers_into_existing_command_buffer(*commandBuffer));
			commandBuffer->end_recording();
			return;
		}
#endif

		compute_frustum_culling(commandBuffer, fif);	// calculate frustum visibility

#if ENABLE_SHADOWMAP
		if (mShadowMap.enable) {
			rdoc::beginSection(commandBuffer->handle(), "Shadowmap", fif);
			// TODO - move shadowmap creation into a subpass?

			for (int cascade = 0; cascade < mShadowMap.numCascades; ++cascade) {
				// Prepare scene draw buffers for current cascade
				compute_scene_draw_buffers(commandBuffer, fif, cascade);

				// bind descriptors
				commandBuffer->bind_descriptors(mPipelineShadowmapOpaque->layout(), mDescriptorCache.get_or_create_descriptor_sets({
					descriptor_binding(0, 0, mMaterialBuffer),
					descriptor_binding(0, 1, mImageSamplers),
					SCENE_DRAW_DESCRIPTOR_BINDINGS(fif)
					descriptor_binding(1, 0, mMatricesUserInputBuffer[fif]),
					descriptor_binding(1, 1, mLightsourcesBuffer[fif])
					}));

				pushc_dii.mShadowMapCascadeToBuild = cascade;

				// start renderpass
				commandBuffer->begin_render_pass_for_framebuffer(mShadowmapRenderpass, mShadowmapPerCascade[cascade].mShadowmapFramebuffer[fif]);

				// draw the opaque parts of the scene
				commandBuffer->bind_pipeline(const_referenced(mPipelineShadowmapOpaque));
				pushc_dii.mDrawType = 0;
				commandBuffer->push_constants(mPipelineShadowmapOpaque->layout(), pushc_dii);
				draw_scene(commandBuffer, fif, false, cascade);

				// draw dynamic objects
				draw_dynamic_objects(commandBuffer, fif, cascade);

				if (mShadowMap.enableForTransparency) {
					// // draw the transparent parts of the scene
					commandBuffer->bind_pipeline(const_referenced(mPipelineShadowmapTransparent));
					pushc_dii.mDrawType = 1;
					commandBuffer->push_constants(mPipelineShadowmapTransparent->layout(), pushc_dii);
					draw_scene(commandBuffer, fif, true, cascade);
				}

				commandBuffer->end_render_pass();
			}
			rdoc::endSection(commandBuffer->handle());
		}
#endif
		// Prepare scene draw buffers for main camera
		compute_scene_draw_buffers(commandBuffer, fif);

		rdoc::beginSection(commandBuffer->handle(), "Render models", fif);
		helpers::record_timing_interval_start(commandBuffer->handle(), fmt::format("mModelsCommandBuffer{} time", fif));

		// Bind the descriptors for descriptor sets 0 and 1 before starting to render with a pipeline
		commandBuffer->bind_descriptors(firstPipe->layout(), mDescriptorCache.get_or_create_descriptor_sets({ // They must match the pipeline's layout (per set!) exactly.
			descriptor_binding(0, 0, mMaterialBuffer),
			descriptor_binding(0, 1, mImageSamplers),
			SCENE_DRAW_DESCRIPTOR_BINDINGS(fif)
			SHADOWMAP_DESCRIPTOR_BINDINGS(fif)
			descriptor_binding(1, 0, mMatricesUserInputBuffer[fif]),
			descriptor_binding(1, 1, mLightsourcesBuffer[fif])
			}));

		// Draw using our pipeline for the first pass (Initially this is the only
		//   pass. After task 2 has been implemented, this is the G-Buffer pass):
		commandBuffer->bind_pipeline(const_referenced(firstPipe));
		commandBuffer->begin_render_pass_for_framebuffer(firstPipe->get_renderpass(), mFramebuffer[fif]);

		// draw the opaque parts of the scene (in deferred shading: draw transparent parts too, we don't use blending there anyway)
		pushc_dii.mDrawType = 0;
		commandBuffer->push_constants(firstPipe->layout(), pushc_dii);
		draw_scene(commandBuffer, fif, false);

		// draw dynamic objects
		draw_dynamic_objects(commandBuffer, fif);

		// draw the transparent parts of the scene
		pushc_dii.mDrawType = 1;
		commandBuffer->bind_pipeline(const_referenced(FORWARD_RENDERING ? secondPipe : firstPipe));
		commandBuffer->push_constants((FORWARD_RENDERING ? secondPipe : firstPipe)->layout(), pushc_dii);
		draw_scene(commandBuffer, fif, true);

#if !FORWARD_RENDERING
		// Move on to next subpass, synchronizing all data to be written to memory,
		// and to be made visible to the next subpass, which uses it as input.
		commandBuffer->next_subpass();

		// TODO - is it necessary to rebind descriptors (pipes are compatible) ??
		commandBuffer->bind_pipeline(const_referenced(secondPipe));
		commandBuffer->bind_descriptors(secondPipe->layout(), mDescriptorCache.get_or_create_descriptor_sets({ 
			descriptor_binding(0, 0, mMaterialBuffer),
			descriptor_binding(0, 1, mImageSamplers),
			SCENE_DRAW_DESCRIPTOR_BINDINGS(fif)
			SHADOWMAP_DESCRIPTOR_BINDINGS(fif)
			descriptor_binding(1, 0, mMatricesUserInputBuffer[fif]),
			descriptor_binding(1, 1, mLightsourcesBuffer[fif]),
			descriptor_binding(2, 0, mFramebuffer[fif]->image_view_at(1)->as_input_attachment(), shader_type::fragment),
			descriptor_binding(2, 1, mFramebuffer[fif]->image_view_at(2)->as_input_attachment(), shader_type::fragment),
			descriptor_binding(2, 2, mFramebuffer[fif]->image_view_at(3)->as_input_attachment(), shader_type::fragment)
		}));

		auto [quadVertices, quadIndices] = helpers::get_quad_vertices_and_indices();
		commandBuffer->draw_indexed(quadIndices, quadVertices);
#endif

		helpers::record_timing_interval_end(commandBuffer->handle(), fmt::format("mModelsCommandBuffer{} time", fif));

		commandBuffer->next_subpass();

		// draw test image
		if (mTestImageSettings.enabled) {
			rdoc::beginSection(commandBuffer->handle(), "Draw test image");
			commandBuffer->bind_pipeline(const_referenced(mPipelineTestImage));
			commandBuffer->bind_descriptors(mPipelineTestImage->layout(), mDescriptorCache.get_or_create_descriptor_sets({
				descriptor_binding(0, 0, mTestImageSettings.bilinear ? mTestImageSampler_bilinear : mTestImageSampler_nearest),
				descriptor_binding(0, 1, mTestImages[mTestImageSettings.imageId]),
				descriptor_binding(1, 0, mMatricesUserInputBuffer[fif]),
			}));
			const auto& [quadVertices, quadIndices] = helpers::get_quad_vertices_and_indices();
			commandBuffer->draw_indexed(quadIndices, quadVertices);
			rdoc::endSection(commandBuffer->handle());
		}

		// draw camera path
		if (mCameraSpline.draw && mDrawCamPathPositions_valid) {
			rdoc::beginSection(commandBuffer->handle(), "Draw camera path");
			commandBuffer->bind_pipeline(const_referenced(mPipelineDrawCamPath));
			commandBuffer->bind_descriptors(mPipelineDrawCamPath->layout(), mDescriptorCache.get_or_create_descriptor_sets({
				descriptor_binding(1, 0, mMatricesUserInputBuffer[fif]),
				}));
			commandBuffer->draw_vertices(static_cast<uint32_t>(mNumCamPathPositions), 1u, 0u, 0u, mDrawCamPathPositionsBuffer);
			rdoc::endSection(commandBuffer->handle());
		}

#if ENABLE_SHADOWMAP
		// draw frustum
		if (mShadowMap.drawFrustum) {
			rdoc::beginSection(commandBuffer->handle(), "Draw frustum");
			commandBuffer->bind_pipeline(const_referenced(mPipelineDrawFrustum));
			commandBuffer->bind_descriptors(mPipelineDrawFrustum->layout(), mDescriptorCache.get_or_create_descriptor_sets({
				descriptor_binding(1, 0, mMatricesUserInputBuffer[fif]),
				}));
			// there is no commandBuffer->draw_vertices without vertex buffer... call vkCmdDraw directly 
			commandBuffer->handle().draw(12 * 2 * (mEffectiveCamera.detached ? 5 : 4), 1, 0, 0);
			rdoc::endSection(commandBuffer->handle());
		}

		// draw shadowmap
		if (mShadowMap.show) {
			rdoc::beginSection(commandBuffer->handle(), "Draw shadow map");
			commandBuffer->bind_pipeline(const_referenced(mPipelineDrawShadowmap));
			commandBuffer->bind_descriptors(mPipelineDrawShadowmap->layout(), mDescriptorCache.get_or_create_descriptor_sets({
				descriptor_binding(1, 0, mMatricesUserInputBuffer[fif]),
				SHADOWMAP_DESCRIPTOR_BINDINGS(fif)
				descriptor_binding(5, 0, mGenericSamplerNearestNeighbour)
				}));
			const auto& [quadVertices, quadIndices] = helpers::get_quad_vertices_and_indices();
			commandBuffer->draw_indexed(quadIndices, quadVertices);
			rdoc::endSection(commandBuffer->handle());
		}
#endif

		commandBuffer->end_render_pass();
		rdoc::endSection(commandBuffer->handle());

		//if (mDoRayTraceTest) {
		//	blit_image(mRtImageViews[fif]->get_image(), mFramebuffer[fif]->image_view_at(0)->get_image(), sync::with_barriers_into_existing_command_buffer(*commandBuffer));
		//}

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

				static bool showCamPathDefWindow = false;
				static bool showNotepadWindow    = false;
				static bool showImguiDemoWindow  = false;

				static CameraState savedCamState = {};
				if (firstTimeInit) {
					savedCamState = { "saved", mQuakeCam.translation(), mQuakeCam.rotation() };
					mCameraPresets[0].t = savedCamState.t;
					mCameraPresets[0].r = savedCamState.r;
					mEffectiveCamera.storeValuesFrom(mQuakeCam);
				}


				static auto smplr = context().create_sampler(filter_mode::bilinear, border_handling_mode::clamp_to_edge);
				static auto texIdsAndDescriptions = [&]() {

					// Gather all the framebuffer attachments to display them
					std::vector<avk::resource_reference<image_view_t>> pViews;
					for (size_t i = 0; i < mFramebuffer[0]->image_views().size(); ++i) pViews.emplace_back(mFramebuffer[0]->image_view_at(i));
#if ENABLE_SHADOWMAP
					//pViews.push_back(&mShadowmapFramebuffer[0]->image_view_at(0));	// doesn't work, since alpha channel = 0, nothing is displayed
#endif
					std::vector<std::tuple<std::optional<ImTextureID>, std::string>> v;
					for (size_t i = 0; i < pViews.size(); ++i) {
						if (pViews[i]->get_image().config().samples != vk::SampleCountFlagBits::e1) {
							LOG_INFO(fmt::format("Excluding framebuffer attachment #{} from the UI because it has a sample count != 1. Wouldn't be displayed properly, sorry.", i));
							v.emplace_back(std::optional<ImTextureID>{}, fmt::format("Not displaying attachment #{}", i));
						} else {
							if (!is_norm_format(pViews[i]->get_image().config().format) && !is_float_format(pViews[i]->get_image().config().format)) {
								LOG_INFO(fmt::format("Excluding framebuffer attachment #{} from the UI because it has format that can not be sampled with a (floating point-type) sampler2D.", i));
								v.emplace_back(std::optional<ImTextureID>{}, fmt::format("Not displaying attachment #{}", i));
							} else {
								v.emplace_back(
									ImGui_ImplVulkan_AddTexture(smplr->handle(), pViews[i]->handle(), static_cast<VkImageLayout>(vk::ImageLayout::eShaderReadOnlyOptimal)),
									fmt::format("Attachment #{}", i)
								);
							}
						}
					}
					return v;

					//std::vector<std::tuple<std::optional<ImTextureID>, std::string>> v;
					//const auto n = mFramebuffer[0]->image_views().size();
					//for (size_t i = 0; i < n; ++i) {
					//	if (mFramebuffer[0]->image_view_at(i)->get_image().config().samples != vk::SampleCountFlagBits::e1) {
					//		LOG_INFO(fmt::format("Excluding framebuffer attachment #{} from the UI because it has a sample count != 1. Wouldn't be displayed properly, sorry.", i));
					//		v.emplace_back(std::optional<ImTextureID>{}, fmt::format("Not displaying attachment #{}", i));
					//	} else {
					//		if (!is_norm_format(mFramebuffer[0]->image_view_at(i)->get_image().config().format) && !is_float_format(mFramebuffer[0]->image_view_at(i)->get_image().config().format)) {
					//			LOG_INFO(fmt::format("Excluding framebuffer attachment #{} from the UI because it has format that can not be sampled with a (floating point-type) sampler2D.", i));
					//			v.emplace_back(std::optional<ImTextureID>{}, fmt::format("Not displaying attachment #{}", i));
					//		} else {
					//			v.emplace_back(
					//				ImGui_ImplVulkan_AddTexture(smplr->handle(), mFramebuffer[0]->image_view_at(i)->handle(), static_cast<VkImageLayout>(vk::ImageLayout::eShaderReadOnlyOptimal)),
					//				fmt::format("Attachment #{}", i)
					//			);
					//		}
					//	}
					//}
					//return v;

				}();

				auto inFlightIndex = context().main_window()->in_flight_index_for_frame();

				Begin("Info & Settings");
				SetWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
				SetWindowSize(ImVec2(250.0f, 500.0f), ImGuiCond_FirstUseEver);

#if FORWARD_RENDERING
				if (mUseAlphaBlending) {
					TextColored(ImVec4(0.f, .6f, .8f, 1.f), "Forward rendering, alpha blending");
				} else {
					TextColored(ImVec4(0.f, .6f, .8f, 1.f), "Forward rendering, alpha testing");
				}
#else
				TextColored(ImVec4(0.f, .6f, .8f, 1.f), "Deferred shading, alpha testing");
#endif
				if (mUpsampling) {
					TextColored(ImVec4(0.f, .6f, .8f, 1.f), "%g x upsampling %dx%d->%dx%d", mUpsamplingFactor, mLoResolution.x, mLoResolution.y, mHiResolution.x, mHiResolution.y);
				} else {
					TextColored(ImVec4(0.f, .6f, .8f, 1.f), "resolution %dx%d", mLoResolution.x, mLoResolution.y);
				}

				// misc warnings of incompatible settings here
				//if ((mShadowMap.enable || mShadowMap.enableForTransparency) && mSceneData.mRegeneratePerFrame && mSceneData.mCullViewFrustum) {
				//	TextColored(ImVec4(1.f, 0.f, 0.f, 1.f), "WARN: Shadows & frustum culling");
				//}

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



					Combo("Lighting", &mLightingMode, "Blinn-Phong\0Color only\0SM cascade\0Normals\0Normals 2\0Lod\0");
				}

				if (CollapsingHeader("Rendering")) {
#if ENABLE_RAYTRACING
					Checkbox("Ray trace whole scene", &mDoRayTraceTest);
					if (InputIntW(80, "RT samples##RtSamples", &mRtSamplesPerPixel)) mRtSamplesPerPixel = glm::clamp(mRtSamplesPerPixel, 1, RAYTRACING_MAX_SAMPLES_PER_PIXEL);
					Checkbox("RT debug sparse tracing", &mRtDebugSparse); HelpMarker("Show areas to be sparsely traced instead of actually ray tracing them");
					Checkbox("RT approximate Lod", &mRtApproximateLod);
#endif
					SliderFloatW(100, "alpha thresh.", &mAlphaThreshold, 0.f, 1.f, "%.3f", 2.f); HelpMarker("Consider anything with less alpha completely invisible (even if alpha blending is enabled).");
					if (Checkbox("alpha blending", &mUseAlphaBlending)) invalidate_command_buffers();
					HelpMarker("When disabled, simple alpha-testing is used.");
					PushItemWidth(60);
					InputFloat("lod bias", &mLodBias, 0.f, 0.f, "%.1f");
					PopItemWidth();
					SameLine();
					Checkbox("taa only##lod bias taa only", &mLoadBiasTaaOnly);
					Checkbox("always use lod 0", &mAlwaysUseLod0); HelpMarker("To match ray tracing image");
					SliderFloatW(100, "normal mapping", &mNormalMappingStrength, 0.0f, 1.0f);
					if (Button("Re-record commands")) invalidate_command_buffers();
				}

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
					Checkbox("strafing", &mAutoStrafe); SameLine(); InputFloatW(40,"spd##autoStrafeSpeed", &mAutoStrafeSpeed, 0.f, 0.f, "%.1f"); SameLine(); InputFloatW(40,"d##autoStrafeDist", &mAutoStrafeDist, 0.f, 0.f, "%.1f");
					Separator();
					if (Button("save cam")) { savedCamState.t = mQuakeCam.translation(); savedCamState.r = mQuakeCam.rotation(); };
					SameLine();
					if (Button("restore cam")) { mQuakeCam.set_translation(savedCamState.t); mQuakeCam.set_rotation(savedCamState.r); }
					SameLine();
					if (Button("print cam")) {
						glm::vec3 t = mQuakeCam.translation();
						glm::quat r = mQuakeCam.rotation();
						//printf("{ \"name\", {%gf, %gf, %gf}, {%gf, %gf, %gf, %gf} },\n", t.x, t.y, t.z, r.w, r.x, r.y, r.z);
						printf("{ \"name\", {%ff, %ff, %ff}, {%ff, %ff, %ff, %ff} },\n", t.x, t.y, t.z, r.w, r.x, r.y, r.z);
					}

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

					Checkbox("follow path", &mCameraSpline.enable);
					if (mCameraSpline.enable && !mCameraSpline.interpolationCurve.valid()) mCameraSpline.enable = false; // need min. 2 or 4 pts
					if (mCameraSpline.enable && mCameraSpline.tStart == 0.f) mCameraSpline.tStart = static_cast<float>(glfwGetTime());
					SameLine();
					if (Button("reset")) mCameraSpline.tStart = static_cast<float>(glfwGetTime());
					SameLine();
					if (Button("edit")) showCamPathDefWindow = true;
					Checkbox("cycle", &mCameraSpline.cyclic);
					SameLine();
					Checkbox("look along", &mCameraSpline.lookAlong);

					if (Checkbox("detach", &mEffectiveCamera.detached)) invalidate_command_buffers();
					SameLine(); if (Button("set to cam")) mEffectiveCamera.storeValuesFrom(mQuakeCam);

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
					auto &dynObj = mDynObjects[mMovingObject.moverId];
					if (dynObj.mIsAnimated) {
						auto &clip = dynObj.mAnimClips[dynObj.mActiveAnimation];
						float minAnimTime = static_cast<float>(clip.mStartTicks / clip.mTicksPerSecond);
						float maxAnimTime = static_cast<float>(clip.mEndTicks   / clip.mTicksPerSecond);
						SliderFloatW(120, "anim", &dynObj.mAnimTime, minAnimTime, maxAnimTime);
						SameLine(); Checkbox("auto##auto anim", &mMovingObject.autoAnimate);
						SliderFloat("speed##anim speed", &mMovingObject.animSpeed, -2.f, 2.f, "%.1f");
					}
					PopID();

					if (old_enabled != mMovingObject.enabled || old_moverId != mMovingObject.moverId) invalidate_command_buffers();
				}

				if (CollapsingHeader("Debug")) {
					if (Checkbox("image", &mTestImageSettings.enabled)) invalidate_command_buffers();	// requires permanent re-recording anyway due to push constants ... FIXME
					SameLine();
					struct FuncHolder {
						static bool ImgGetter(void* data, int idx, const char** out_str) {
							*out_str = reinterpret_cast<wookiee*>(data)->mImageDefs[idx].name;
							return true;
						}
					};
					if (Combo("##image id", &mTestImageSettings.imageId, &FuncHolder::ImgGetter, this, static_cast<int>(mImageDefs.size()))) invalidate_command_buffers();
					if (Checkbox("bilinear sampling", &mTestImageSettings.bilinear)) invalidate_command_buffers();

					Checkbox("Regen.scene buffers", &mSceneData.mRegeneratePerFrame);
					Checkbox("Cull view frustum",   &mSceneData.mCullViewFrustum);

					if (rdoc::active()) {
						Separator();
						if (Button("capture") && !mCaptureFramesLeft && mCaptureNumFrames > 0) mStartCapture = true;
						SameLine();
						PushItemWidth(60);
						InputInt("frames", &mCaptureNumFrames);
						PopItemWidth();
					}

					if (Button("Show ImGui demo window")) showImguiDemoWindow = true;
				}

				if (CollapsingHeader("Shadows")) {
					PushID("ShadowStuff");
					if (Checkbox("enable", &mShadowMap.enable)) invalidate_command_buffers();
					SameLine();
					if (Checkbox("transp.", &mShadowMap.enableForTransparency)) invalidate_command_buffers();

					if (Checkbox("show shadowmap", &mShadowMap.show)) invalidate_command_buffers();
					if (Checkbox("show frustum", &mShadowMap.drawFrustum)) invalidate_command_buffers();

					Checkbox("restrict to scene", &mShadowMap.shadowMapUtil.restrictLightViewToScene);
					SliderInt("num cascades", &mShadowMap.desiredNumCascades, 1, SHADOWMAP_MAX_CASCADES);
					PushItemWidth(40);
					Text("Casc:");
					for (int i = 0; i < mShadowMap.numCascades; ++i) {
						PushID(i);
						SameLine(); InputFloat("##cascend", &mShadowMap.shadowMapUtil.cascadeEnd[i], .0f, .0f, "%.2f");
						PopID();
					}
					SameLine();
					if (Checkbox("auto##autocascade", &mShadowMap.autoCalcCascadeEnds) && mShadowMap.autoCalcCascadeEnds) mShadowMap.shadowMapUtil.calc_cascade_ends();
					PopItemWidth();

					InputFloat("manual bias", &mShadowMap.bias);
					Text("Depth bias (constant, slope, clamp):");
					bool mod = false;
					for (int i = 0; i < mShadowMap.numCascades; ++i) {
						PushID(i);
						PushItemWidth(60);
						Text("%d", i); SameLine();
						mod = mod || Checkbox("##dbenable", &mShadowMap.depthBias[i].enable);						SameLine();
						mod = mod || InputFloat("##dbconst", &mShadowMap.depthBias[i].constant, .0f, .0f, "%.2f");	SameLine();
						mod = mod || InputFloat("##dbslope", &mShadowMap.depthBias[i].slope,    .0f, .0f, "%.2f");	SameLine();
						mod = mod || InputFloat("##dbclamp", &mShadowMap.depthBias[i].clamp,    .0f, .0f, "%.2f");
						PopItemWidth();
						PopID();
					}
					if (mod) invalidate_command_buffers();

					PopID();
				}

				if (CollapsingHeader("Result images")) {
					for (auto& tpl : texIdsAndDescriptions) {
						auto texId = std::get<std::optional<ImTextureID>>(tpl);
						auto description = std::get<std::string>(tpl);
						if (texId.has_value()) {
							Image(texId.value(), ImVec2(192, 108)); SameLine();
						}
						Text(description.c_str());
					}
				}

				static std::string lastSettingsFn = "settings.ini";

				Separator();
				Text("Settings:"); SameLine();
				if (Button("Load...")) {
					//if (lastFn == "") lastFn = std::experimental::filesystem::canonical(gAssetsDir + "/params/").string();
					auto fns = pfd::open_file("Load settings", lastSettingsFn, { "Setting files (.ini)", "*.ini", "All files", "*" }).result();
					if (fns.size()) {
						std::string oldNotes = mNotepad;
						loadSettings(fns[0]);
						lastSettingsFn = fns[0];
						if (mNotepad != "" && mNotepad != oldNotes) showNotepadWindow = true;
					}
				}
				SameLine();
				if (Button("Save...")) {
					//if (lastParamsFn == "") lastParamsFn = std::experimental::filesystem::canonical(gAssetsDir + "/params/").string();
					auto fn = pfd::save_file("Save settings as", lastSettingsFn, { "Setting files (.ini)", "*.ini", "All files", "*" }).result();
					if (fn != "") {
						saveSettings(fn, true);
						lastSettingsFn = fn;
					}
				}
				if (Button("Notepad##ShowNotepad")) showNotepadWindow = !showNotepadWindow;

				End();	// main window

				// ----- notepad -----------------------------------------------------
				if (showNotepadWindow) {
					Begin("Notepad", &showNotepadWindow);
					SetWindowPos (ImVec2( 10.0f, 550.0f), ImGuiCond_FirstUseEver);		// TODO: fix pos
					SetWindowSize(ImVec2(500.0f, 500.0f), ImGuiCond_FirstUseEver);
					InputTextMultiline("##notes", &mNotepad, ImVec2(-FLT_MIN, -FLT_MIN));
					End();
				}

				// ----- camera path def window --------------------------------------
				if (showCamPathDefWindow) {
					Begin("Camera path", &showCamPathDefWindow);
					SetWindowPos (ImVec2(500.0f,  10.0f), ImGuiCond_FirstUseEver);		// TODO: fix pos
					SetWindowSize(ImVec2(250.0f, 400.0f), ImGuiCond_FirstUseEver);

					bool changed = false;
					static std::string lastPathEditorFn = "";

					static std::vector<glm::vec3> pos = mCameraSpline.interpolationCurve.control_points();

					if (!mCameraSpline.editorControlPointsValid) {
						pos = mCameraSpline.interpolationCurve.control_points();
						mCameraSpline.editorControlPointsValid = true;
					}

					//static std::vector<glm::quat> rot; // not used for now
					//if (rot.size() != pos.size()) rot.resize(pos.size(), glm::quat(1, 0, 0, 0));

					bool isCatmull = mCameraSpline.interpolationCurve.type() == InterpolationCurveType::catmull_rom_spline;

					auto cameraT = mQuakeCam.translation();
					auto cameraR = mQuakeCam.rotation();


					if (Button("Clear")) { pos.clear(); pos.push_back(glm::vec3(0)); changed = true; }

					SameLine();
					if (Button("Load...")) {
						//if (lastFn == "") lastFn = std::experimental::filesystem::canonical(gAssetsDir + "/params/").string();
						auto fns = pfd::open_file("Load camera path", lastPathEditorFn, { "Cam path files (.cam)", "*.cam", "All files", "*" }).result();
						if (fns.size()) {
							loadCamPath(fns[0]);
							pos = mCameraSpline.interpolationCurve.control_points();
							changed = true;
						}
					}
					SameLine();
					if (Button("Save...")) {
						//if (lastParamsFn == "") lastParamsFn = std::experimental::filesystem::canonical(gAssetsDir + "/params/").string();
						auto fn = pfd::save_file("Save camera path as", lastPathEditorFn, { "Cam path files (.cam)", "*.cam", "All files", "*" }).result();
						if (fn != "") {
							saveCamPath(fn, true);
							lastPathEditorFn = fn;
						}
					}

					InputFloatW(60, "duration", &mCameraSpline.duration, 0.f, 0.f, "%.1f");
					SameLine();
					if (Checkbox("const.speed", &mCameraSpline.constantSpeed)) mDrawCamPathPositions_valid = false;
					SameLine();
					//float alpha = mCameraSpline.spline.get_catmullrom_alpha();
					//if (InputFloatW(40, "a##Catmull-Rom alpha", &alpha, 0.f, 0.f, "%.1f")) { mCameraSpline.spline.set_catmullrom_alpha(alpha); changed = true; }
					//HelpMarker("Catmull-Rom alpha (affects path interpolation):\n0.5: centripetal\n0.0: uniform\n1.0: chordal");

					//ComboW(120, "rotate", &mCameraSpline.spline.rotation_mode, "none\0from keyframes\0from position\0"); // FIXME
					Checkbox("look along", &mCameraSpline.lookAlong);
					//SameLine();
					//static bool show_rot = true;
					//Checkbox("show rotation input", &show_rot);
					int delPos = -1;
					int addPos = -1;
					int moveUp = -1;
					int moveDn = -1;
					int numP = static_cast<int>(pos.size());

					int splineType = static_cast<int>(mCameraSpline.interpolationCurve.type());
					if (Combo("type##splinetype", &splineType, "Bezier\0Quad. B-spline\0Cubic B-spline\0Catmull-Rom\0")) {
						mCameraSpline.interpolationCurve.setType(static_cast<InterpolationCurveType>(splineType));
						mDrawCamPathPositions_valid = false;
					}

					Separator();
					BeginChild("scrollbox", ImVec2(0, 200));
					for (int i = 0; i < numP; ++i) {
						PushID(i);
						if (isCatmull && (i == 1 || i == numP - 1)) Separator();
						if (InputFloat3W(120, "##pos", &(pos[i].x), "%.2f")) changed = true;
						//if (show_rot) {
						//	SameLine();
						//	if (InputFloat4W(160, "##rot", &(rot[i].x), "%.2f")) changed = true;
						//}
						SameLine(); if (Button("set")) { pos[i] = mQuakeCam.translation(); /* rot[i] = mQuakeCam.rotation(); changed = true; */ }
						SameLine(); if (Button("-")) delPos = i;
						SameLine(); if (Button("+")) addPos = i;
						SameLine(); if (Button("^")) moveUp = i;
						SameLine(); if (Button("v")) moveDn = i;
						SameLine(); if (Button("go")) { mQuakeCam.set_translation(pos[i]); /* mQuakeCam.set_rotation(rot[i]); */ }
						PopID();
					}
					EndChild();

					if (addPos >= 0)											{ pos.insert(pos.begin() + addPos + 1, cameraT);		/* rot.insert(rot.begin() + addPos + 1, cameraR);	*/	changed = true; }
					if (delPos >= 0)											{ pos.erase (pos.begin() + delPos);						/* rot.erase (rot.begin() + delPos);				*/	changed = true; }
					if (moveUp >  0)											{ std::swap(pos[moveUp - 1], pos[moveUp]);				/* std::swap(rot[moveUp - 1], rot[moveUp]);			*/	changed = true; }
					if (moveDn >= 0 && moveDn < static_cast<int>(pos.size())-1) { std::swap(pos[moveDn + 1], pos[moveDn]);				/* std::swap(rot[moveDn + 1], rot[moveDn]);			*/	changed = true; }

					Separator();
					auto t = cameraT;
					auto r = cameraR;
					//if (show_rot)	Text("cam pos %.2f %.2f %.2f  rot %.2f %.2f %.2f %.2f",	t.x, t.y, t.z, r.x, r.y, r.z, r.w);
					//else			Text("cam pos %.2f %.2f %.2f",							t.x, t.y, t.z);
					Text("cam pos %.2f %.2f %.2f",							t.x, t.y, t.z);
					Separator();

					if (isCatmull) {
						if (Button("auto-set cycle lead in/out")) {
							// affects last regular point and in/out points
							auto sz = pos.size();
							if (sz > 3) {
								pos[sz - 2] = pos[1];	/* rot[sz - 2] = rot[1]; */
								pos[sz - 1] = pos[2];	/* rot[sz - 1] = rot[2]; */
								pos[0] = pos[sz - 3];	/* rot[0] = rot[sz - 3]; */
								changed = true;
							}
						}
						HelpMarker("Update points for cyclic path:\nLast regular point is set to first regular point.\nLead-in and lead-out are set to match cycle.");
					}

					Checkbox("interactive editor", &mCameraSpline.draw);
					HelpMarker("Drag control points in XZ plane.\nHold down SHIFT to change Y.");
					SameLine();
					if (InputIntW(80, "points", &mCameraSpline.drawNpoints, 0)) mDrawCamPathPositions_valid = false;
					if (mCameraSpline.drawNpoints > mMaxCamPathPositions) mCameraSpline.drawNpoints = mMaxCamPathPositions;


					if (changed) {
						// mCameraSpline.spline.modified(); // FIXME - recalc arclen?
						mDrawCamPathPositions_valid = false;
						mCameraSpline.interpolationCurve.set_control_points(pos);
					} 

					End();
				}

				
				if (showImguiDemoWindow) ShowDemoWindow(&showImguiDemoWindow);

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

		if (mStartCaptureEarly) {
			mStartCapture = false;
			mCaptureFramesLeft = mCaptureNumFrames;
			rdoc::start_capture();
			LOG_INFO("Starting early capture for " + std::to_string(mCaptureNumFrames) + " frames");
		}

		auto* wnd = context().main_window();

		mHiResolution = wnd->resolution();
		mLoResolution = mUpsampling ? glm::uvec2(glm::vec2(mHiResolution) / mUpsamplingFactor) : mHiResolution;

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
		prepare_shadowmap();

		// create an all-purpose nearest neighbour sampler
		mGenericSamplerNearestNeighbour = gvk::context().create_sampler(avk::filter_mode::nearest_neighbor, avk::border_handling_mode::clamp_to_edge);

		// create a buffer for drawing camera path
		mDrawCamPathPositionsBuffer = gvk::context().create_buffer(avk::memory_usage::device, {}, avk::vertex_buffer_meta::create_from_element_size(sizeof(glm::vec4), mMaxCamPathPositions).describe_only_member(glm::vec4(0), avk::content_description::position));

		load_and_prepare_scene();

#if FORWARD_RENDERING
		prepare_forward_rendering_pipelines();
#else
		prepare_deferred_shading_pipelines();
#endif
		prepare_common_pipelines();

		// print_pipelines_info();

#if ENABLE_RAYTRACING
		init_raytracing();
#endif

		// alloc command buffers for drawing the scene, but don't record them yet
		alloc_command_buffers_for_models();
		invalidate_command_buffers();

		// Add the camera to the composition (and let it handle the updates)
		mQuakeCam.set_translation({ 0.0f, 1.0f, 0.0f });
		//mQuakeCam.set_perspective_projection(glm::radians(60.0f), context().main_window()->aspect_ratio(), 0.1f, 500.0f);
		// set far plane to ~diagonal of scene bounding box
		float diagonal = glm::distance(mSceneData.mBoundingBox.min, mSceneData.mBoundingBox.max);
		mQuakeCam.set_perspective_projection(glm::radians(60.0f), context().main_window()->aspect_ratio(), 0.1f, diagonal);
		mOriginalProjMat = mQuakeCam.projection_matrix();
		current_composition()->add_element(mQuakeCam);

		// re-init shadow map (and calc cascades), after we have camera near/far plane
		re_init_shadowmap();

		// load default camera path
		if (!loadCamPath("assets/defaults/camera_path.cam")) LOG_WARNING("Failed to load default camera path");

		setup_ui_callback();

		upload_materials_and_vertex_data_to_gpu();

		std::array<image_view_t*, cConcurrentFrames> srcDepthImages;
//		std::array<image_view_t*, cConcurrentFrames> srcUvNrmImages;
		std::array<image_view_t*, cConcurrentFrames> srcMatIdImages;
		std::array<image_view_t*, cConcurrentFrames> srcColorImages;
		std::array<image_view_t*, cConcurrentFrames> srcVelocityImages;
		std::array<image_view_t*, cConcurrentFrames> srcRayTracedImages;
		auto fif = wnd->number_of_frames_in_flight();
		for (decltype(fif) i = 0; i < fif; ++i) {
			srcDepthImages[i]    = &mFramebuffer[i]->image_view_at(1).get();
//			srcUvNrmImages[i]    = &mFramebuffer[i]->image_view_at(2).get();
			srcMatIdImages[i]    = &mFramebuffer[i]->image_view_at(FORWARD_RENDERING ? 2 : 3).get();
			srcColorImages[i]    = &mFramebuffer[i]->image_view_at(0).get();
			srcVelocityImages[i] = &mFramebuffer[i]->image_view_at(FORWARD_RENDERING ? 3 : 4).get();
#if ENABLE_RAYTRACING
			srcRayTracedImages[i] = &mRtImageViews[i].get();
#else
			srcRayTracedImages[i] = nullptr;
#endif
		}

		mAntiAliasing.set_source_image_views(mHiResolution, srcColorImages, srcDepthImages, srcVelocityImages, srcMatIdImages, srcRayTracedImages);
		mAntiAliasing.register_raytrace_callback(this);
		current_composition()->add_element(mAntiAliasing);

		init_debug_stuff();

		if (mHideWindowOnLoad) glfwShowWindow(glfwWin);
	}

	void init_debug_stuff() {
		return;
		mMovingObject.moverId = 3;
		mMovingObject.startPos = glm::vec3(0,0,0);
		mMovingObject.endPos   = glm::vec3(0,0,0);
		mMovingObject.rotAxisAngle.w = 0.f;
		mMovingObject.autoAnimate = true;
		mMovingObject.enabled  = true;
	}

	void init_updater_only_once() {
		static bool beenThereDoneThat = false;
		if (beenThereDoneThat) return;
		beenThereDoneThat = true;

		mUpdater.emplace();

		// allow the updater to reload shaders
#if USE_GVK_UPDATER
		mAntiAliasing.init_updater();

		// updating the pipelines crashes when using pre-recorded command buffers! (because pipeline is still in use)
#if RERECORD_CMDBUFFERS_ALWAYS

		#if FORWARD_RENDERING
			std::vector<avk::graphics_pipeline *> gfx_pipes = { &mPipelineFwdOpaque, &mPipelineFwdTransparent, &mPipelineFwdTransparentNoBlend };
		#else
			std::vector<avk::graphics_pipeline *> gfx_pipes = { &mPipelineFirstPass, &mPipelineLightingPass };
		#endif
		gfx_pipes.push_back(&mPipelineAnimObject);
		gfx_pipes.push_back(&mPipelineTestImage);
		gfx_pipes.push_back(&mPipelineDrawCamPath);
		#if ENABLE_SHADOWMAP
			gfx_pipes.push_back(&mPipelineShadowmapOpaque);
			gfx_pipes.push_back(&mPipelineShadowmapTransparent);
		#endif

		for (auto ppipe : gfx_pipes) {
			ppipe->enable_shared_ownership();
			mUpdater->on(gvk::shader_files_changed_event(*ppipe)).update(*ppipe);
		}
		#if ENABLE_RAYTRACING
			std::vector<avk::ray_tracing_pipeline *> rtx_pipes = { &mPipelineRayTrace };
			for (auto ppipe : rtx_pipes) {
				ppipe->enable_shared_ownership();
				mUpdater->on(gvk::shader_files_changed_event(*ppipe)).update(*ppipe);
			}
		#endif

#endif
		//gvk::current_composition()->add_element(mUpdater);
#endif
	}

	void handle_input_for_path_editor() {
		using namespace gvk;

		if (!mCameraSpline.draw) return;
		if (mQuakeCam.is_enabled()) return;

		ImGuiIO &io = ImGui::GetIO();
		if (io.WantCaptureMouse) return;

		static glm::ivec2 prev_pos = {0,0};
		static bool prev_lmb = false;
		static int prev_cpt_id = -1;

		const bool restrictEditAxes = true; // better to design floor aligned paths: change only xz (unless shift is pressed, then change y)

		glm::ivec2 pos = glm::ivec2(input().cursor_position());
		bool lmb = (input().mouse_button_down(GLFW_MOUSE_BUTTON_LEFT));
		bool shift = input().key_down(key_code::left_shift) || input().key_down(key_code::right_shift);

		bool click = lmb && !prev_lmb;
		bool drag  = lmb && prev_lmb;

		glm::mat4 pvMat = mQuakeCam.projection_and_view_matrix();

		auto world_to_ndc = [&pvMat](glm::vec3 p) -> glm::vec3 {
			glm::vec4 ndc = pvMat * glm::vec4(p, 1);
			if (abs(ndc.w) > 0.00001f) ndc /= ndc.w;
			return ndc;
		};

		auto ndc_to_screen = [this](glm::vec3 ndc) -> glm::vec2 {
			glm::vec2 xy = glm::vec2(ndc);
			xy = xy * 0.5f + 0.5f;
			return xy * glm::vec2(mHiResolution);
		};


		int cpt_id = prev_cpt_id;
		if (!drag) {
			// test if we are above a control point

			cpt_id = -1;
			const float minDistForHit = 5.f;
			float closest_distance = std::numeric_limits<float>::max();
			auto &cpts = mCameraSpline.interpolationCurve.control_points();
			for (size_t i = 0; i < mCameraSpline.interpolationCurve.num_control_points(); ++i) {
				glm::vec3 ndc = world_to_ndc(mCameraSpline.interpolationCurve.control_point_at(i));
				float d = glm::length(glm::vec2(pos) - ndc_to_screen(ndc));
				if (d < minDistForHit && d < closest_distance) {
					closest_distance = d;
					cpt_id = static_cast<int>(i);
				}
			}
			context().main_window()->set_cursor_mode((cpt_id >= 0) ? ((restrictEditAxes && shift) ? cursor::vertical_resize_cursor : cursor::crosshair_cursor) : cursor::arrow_cursor);
		}

		if (cpt_id >= static_cast<int>(mCameraSpline.interpolationCurve.num_control_points())) cpt_id = -1;

		if (lmb && cpt_id >= 0) {
			// keep depth
			glm::vec3 cpt_orig = mCameraSpline.interpolationCurve.control_point_at(cpt_id);
			glm::vec3 cpt_ndc = world_to_ndc(cpt_orig);
			glm::vec4 ndc = glm::vec4((glm::vec2(pos) / glm::vec2(mHiResolution)) * 2.f - 1.f, cpt_ndc.z, 1);
			glm::vec4 world = glm::inverse(pvMat) * ndc;
			if (abs(world.w) > 0.00001f) world /= world.w;

			// restrict
			if (restrictEditAxes) {
				if (!shift) {
					world.y = cpt_orig.y;
				} else {
					world.x = cpt_orig.x;
					world.z = cpt_orig.z;
				}
			}

			//printf("new %f %f %f\n", world.x, world.y, world.z);
			std::vector<glm::vec3> cpts = mCameraSpline.interpolationCurve.control_points();
			cpts[cpt_id] = world;
			mCameraSpline.interpolationCurve.set_control_points(cpts);
			mCameraSpline.editorControlPointsValid = false;
			mDrawCamPathPositions_valid = false;
		}

		prev_pos = pos;
		prev_lmb = lmb;
		prev_cpt_id = cpt_id;

	}

	void update() override
	{
		// NOTE: Do NOT write GPU resources here that were used #concurrentFrames before. These commands may still be executing.
		//       Do it in render(), there it is guaranteed that frame #(thisFrame - concurrentFrames) has finished.

		init_updater_only_once(); // don't init in initialize, taa hasn't created its pipelines there yet.

		handle_input_for_path_editor();

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

		// camera path
		if (mCameraSpline.enable) {
			// FIXME for slow-mo! also need to fix reset
			float t_spline = (t - mCameraSpline.tStart) / mCameraSpline.duration;
			if (t_spline > 1.f) {
				if (mCameraSpline.cyclic) t_spline = glm::fract(t_spline); else { mCameraSpline.enable = false; mCameraSpline.tStart = 0.f; t_spline = 1.f; }
			}
			//glm::vec3 pos = mQuakeCam.translation();
			//glm::quat rot = mQuakeCam.rotation();
			//mCameraSpline.spline.interpolate(t_spline, pos, rot);
			//mQuakeCam.set_translation(pos);
			//mQuakeCam.set_rotation(rot);
			if (mCameraSpline.constantSpeed) t_spline = mCameraSpline.interpolationCurve.mapConstantSpeedTime(t_spline);
			mQuakeCam.set_translation(mCameraSpline.interpolationCurve.value_at(t_spline));
			if (mCameraSpline.lookAlong) mQuakeCam.look_along(mCameraSpline.interpolationCurve.slope_at(t_spline));
		}

		float dtCam = dt;
		static float tCamBob = 0.f;
		static float tCamStrafe = 0.f;
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
		static glm::vec3 strafeStart = mQuakeCam.translation();
		if (mAutoMovement && mAutoStrafe) {
			tCamStrafe += dtCam;
			float dx =  mAutoStrafeDist * sin(tCamStrafe * mAutoStrafeSpeed);
			mQuakeCam.set_translation(strafeStart + glm::vec3(dx, 0, 0));
		} else {
			strafeStart = mQuakeCam.translation();
			tCamStrafe = 0.f;
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

			auto &dynObj = mDynObjects[mMovingObject.moverId];
			if (dynObj.mIsAnimated && mMovingObject.autoAnimate) {
				auto &clip = dynObj.mAnimClips[dynObj.mActiveAnimation];
				float minAnimTime = static_cast<float>(clip.mStartTicks / clip.mTicksPerSecond);
				float maxAnimTime = static_cast<float>(clip.mEndTicks   / clip.mTicksPerSecond);
				float timePassed = fmod(abs(mMovingObject.animSpeed) * tObjAccumulated, maxAnimTime - minAnimTime);
				dynObj.mAnimTime = (mMovingObject.animSpeed < 0.f) ? maxAnimTime - timePassed : minAnimTime + timePassed;
			}

			// set current and previous movement matrix
			// FIXME - first time init of prev, init after object was inactive...
			mDynObjects[mMovingObject.moverId].mMovementMatrix_prev    = mDynObjects[mMovingObject.moverId].mMovementMatrix_current;
			mDynObjects[mMovingObject.moverId].mMovementMatrix_current = glm::rotate(glm::translate(glm::mat4(1), mMovingObject.translation), mMovingObject.rotationAngle, glm::vec3(mMovingObject.rotAxisAngle));

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
		auto inFlightIndex = mainWnd->in_flight_index_for_frame();

		#if ADDITIONAL_FRAME_SYNC_WITH_FENCES
			// wait until the frame-fence is signaled
			if (mModelsCommandBufferFence[inFlightIndex].has_value()) mModelsCommandBufferFence[inFlightIndex]->wait_until_signalled();
		#endif

		#if ENABLE_SHADOWMAP
			// re-init shadowmap if numCascades changed
			if (mShadowMap.numCascades != mShadowMap.desiredNumCascades) {
				re_init_shadowmap();
				invalidate_command_buffers();
			}
		#endif

		#if !ENABLE_GPU_FRUSTUM_CULLING	
			// rebuild scene buffers (and perform frustum culling) (when not using GPU-culling)
			if (mSceneData.mRegeneratePerFrame) rebuild_scene_buffers(inFlightIndex); // no command buffer invalidation necessary
		#endif

		update_matrices_and_user_input();
		update_lightsources();
		update_bone_matrices();
		update_culling_ubo();
		update_camera_path_draw_buffer();

#if ENABLE_RAYTRACING
		// after updating bone matrices!
		// update the ray tracing acceleration structures if necessary
		if (mDoRayTraceTest || mAntiAliasing.needRayTraceAssist()) {
			update_top_level_acceleration_structure();		// first! - rebuild TLAS if dynObj was changed!
			update_bottom_level_acceleration_structures();	// second! - will only update TLAS due to AABB changes (and that might go away)
		}
#endif

		#if RERECORD_CMDBUFFERS_ALWAYS
			mModelsCommandBuffer[inFlightIndex]->prepare_for_reuse(); // this basically does: mPostExecutionHandler.reset()
			// note: vkResetCommandBuffer is not necessary - vkBeginCommandBuffer (when re-recording) does that implicitly
			record_single_command_buffer_for_models(mModelsCommandBuffer[inFlightIndex], inFlightIndex);
		#else
			// re-record command buffers for ALL frames-in-flight, but only if necessary (this should only be in response to gui selections)
			if (mReRecordCommandBuffers) {
				gvk::context().device().waitIdle(); // this is ok in THIS application, not generally though!
				record_all_command_buffers_for_models();
				mReRecordCommandBuffers = false;
			}
		#endif

		mQueue->submit(mSkyboxCommandBuffer[inFlightIndex], std::optional<avk::resource_reference<avk::semaphore_t>>{});

		#if ADDITIONAL_FRAME_SYNC_WITH_FENCES
			mModelsCommandBufferFence[inFlightIndex] = mQueue->submit_with_fence(mModelsCommandBuffer[inFlightIndex]);
		#else
			mQueue->submit(mModelsCommandBuffer[inFlightIndex], std::optional<avk::resource_reference<avk::semaphore_t>>{});
		#endif

		// anti_asiasing::render() will be invoked after this
	}

	void finalize() override
	{
		helpers::clean_up_timing_resources();
		gvk::current_composition()->remove_element(mQuakeCam);
	}

	bool saveCamPath(std::string fullFilename, bool overwrite) {
		if (!overwrite) {
			struct stat buffer;
			if (stat(fullFilename.c_str(), &buffer) == 0) return false;
		}

		mINI::INIFile file(fullFilename);
		mINI::INIStructure ini;
		writeCamPathToIni(ini);
		return file.generate(ini);
	}

	bool loadCamPath(std::string fullFilename) {
		mINI::INIFile file(fullFilename);
		mINI::INIStructure ini;

		if (!file.read(ini)) return false;
		readCamPathFromIni(ini);
		return true;
	}

	void writeCamPathToIni(mINI::INIStructure &ini) {
		std::string sec = "CamPath";
		iniWriteInt     (ini, sec, "type",			static_cast<int>(mCameraSpline.interpolationCurve.type()));
		iniWriteFloat	(ini, sec, "duration",		mCameraSpline.duration);
		iniWriteBool	(ini, sec, "constantSpeed",	mCameraSpline.constantSpeed);
		iniWriteBool	(ini, sec, "lookAlong",		mCameraSpline.lookAlong);
		//iniWriteFloat	(ini, sec, "cmr_alpha",	mCameraSpline.spline.get_catmullrom_alpha());

		auto &pos = mCameraSpline.interpolationCurve.control_points();
		iniWriteInt		(ini, sec, "num_pos",	(int)pos.size());
		for (int i = 0; i < (int)pos.size(); ++i) {
			iniWriteVec3(ini, sec, "pos_" + std::to_string(i), pos[i]);
			//iniWriteQuat(ini, sec, "rot_" + std::to_string(i), rot[i]);
		}

	}

	void readCamPathFromIni(mINI::INIStructure &ini) {
		std::string sec = "CamPath";

		int aType = static_cast<int>(mCameraSpline.interpolationCurve.type());
		iniReadInt(ini, sec, "type", aType);	mCameraSpline.interpolationCurve.setType(static_cast<InterpolationCurveType>(aType));
		iniReadFloat(ini, sec, "duration", mCameraSpline.duration);
		iniReadBool(ini, sec, "constantSpeed", mCameraSpline.constantSpeed);
		iniReadBool(ini, sec, "lookAlong", mCameraSpline.lookAlong);
		//float alpha = 0.5f;
		//iniReadFloat	(ini, sec, "cmr_alpha",	alpha); mCameraSpline.spline.set_catmullrom_alpha(alpha);

		int num_pos = 0;
		iniReadInt(ini, sec, "num_pos", num_pos);
		std::vector<glm::vec3> pos(num_pos);
		for (int i = 0; i < num_pos; ++i) {
			iniReadVec3(ini, sec, "pos_" + std::to_string(i), pos[i]);
			//iniReadQuat (ini, sec, "rot_" + std::to_string(i), rot[i]);
		}

		mCameraSpline.interpolationCurve.set_control_points(pos);
		mCameraSpline.editorControlPointsValid = false;
		mDrawCamPathPositions_valid = false;
	}

	bool saveSettings(std::string fullFilename, bool overwrite) {
		if (!overwrite) {
			struct stat buffer;
			if (stat(fullFilename.c_str(), &buffer) == 0) return false;
		}

		mINI::INIFile file(fullFilename);
		mINI::INIStructure ini;
		std::string sec;

		sec = "Lights";
		iniWriteInt		(ini, sec, "mMaxPointLightCount",			mMaxPointLightCount);
		iniWriteInt		(ini, sec, "mMaxSpotLightCount",			mMaxSpotLightCount);
		iniWriteVec3	(ini, sec, "mDirLight.intensity",			mDirLight.intensity);
		iniWriteVec3	(ini, sec, "mDirLight.dir",					mDirLight.dir);
		iniWriteFloat	(ini, sec, "mDirLight.boost",				mDirLight.boost);
		iniWriteVec3	(ini, sec, "mAmbLight.col",					mAmbLight.col);
		iniWriteFloat	(ini, sec, "mAmbLight.boost",				mAmbLight.boost);
		iniWriteInt		(ini, sec, "mLightingMode",					mLightingMode);

		sec = "Rendering";
		iniWriteFloat	(ini, sec, "mAlphaThreshold",				mAlphaThreshold);
		iniWriteBool	(ini, sec, "mUseAlphaBlending",				mUseAlphaBlending);
		iniWriteFloat	(ini, sec, "mLodBias",						mLodBias);
		iniWriteBool	(ini, sec, "mLoadBiasTaaOnly",				mLoadBiasTaaOnly);
		iniWriteBool	(ini, sec, "mAlwaysUseLod0",				mAlwaysUseLod0);
		iniWriteFloat	(ini, sec, "mNormalMappingStrength",		mNormalMappingStrength);

		sec = "Camera";
		iniWriteVec3	(ini, sec, "mQuakeCam.translation",			mQuakeCam.translation());
		iniWriteQuat	(ini, sec, "mQuakeCam.rotation",			mQuakeCam.rotation());
		iniWriteBool	(ini, sec, "mAutoMovement",					mAutoMovement);
		iniWriteInt		(ini, sec, "mAutoMovementUnits",			mAutoMovementUnits);
		iniWriteBool	(ini, sec, "mAutoRotate",					mAutoRotate);
		iniWriteVec2	(ini, sec, "mAutoRotateDegrees",			mAutoRotateDegrees);
		iniWriteBool	(ini, sec, "mAutoBob",						mAutoBob);
		iniWriteBool	(ini, sec, "mAutoStrafe",					mAutoStrafe);
		iniWriteFloat	(ini, sec, "mAutoStrafeSpeed",				mAutoStrafeSpeed);
		iniWriteFloat	(ini, sec, "mAutoStrafeDist",				mAutoStrafeDist);
		iniWriteBool	(ini, sec, "mCameraSpline.enable",			mCameraSpline.enable);	// ATTN on load
		iniWriteBool	(ini, sec, "mCameraSpline.cyclic",			mCameraSpline.cyclic);
		//iniWriteBool	(ini, sec, "mCameraSpline.lookAlong",		mCameraSpline.lookAlong);	// this is stored with cam path

		sec = "DynObjects";
		iniWriteBool	(ini, sec, "mMovingObject.enabled",			mMovingObject.enabled);
		iniWriteInt		(ini, sec, "mMovingObject.moverId",			mMovingObject.moverId);
		iniWriteVec3	(ini, sec, "mMovingObject.startPos",		mMovingObject.startPos);
		iniWriteVec3	(ini, sec, "mMovingObject.endPos",			mMovingObject.endPos);
		iniWriteVec4	(ini, sec, "mMovingObject.rotAxisAngle",	mMovingObject.rotAxisAngle);
		iniWriteFloat	(ini, sec, "mMovingObject.speed",			mMovingObject.speed);
		iniWriteInt		(ini, sec, "mMovingObject.units",			mMovingObject.units);
		iniWriteInt		(ini, sec, "mMovingObject.repeat",			mMovingObject.repeat);
		iniWriteBool	(ini, sec, "mMovingObject.rotContinous",	mMovingObject.rotContinous);
		iniWriteFloat	(ini, sec, "dynObj.mAnimTime",				mDynObjects[mMovingObject.moverId].mAnimTime);
		iniWriteBool	(ini, sec, "mMovingObject.autoAnimate",		mMovingObject.autoAnimate);
		iniWriteFloat	(ini, sec, "mMovingObject.animSpeed",		mMovingObject.animSpeed);

		sec = "Images";
		iniWriteBool	(ini, sec, "mTestImageSettings.enabled",	mTestImageSettings.enabled);
		iniWriteInt		(ini, sec, "mTestImageSettings.imageId",	mTestImageSettings.imageId);
		iniWriteBool	(ini, sec, "mTestImageSettings.bilinear",	mTestImageSettings.bilinear);

		sec = "Shadows";
		iniWriteBool	(ini, sec, "enable",						mShadowMap.enable);
		iniWriteBool	(ini, sec, "enableForTransparency",			mShadowMap.enableForTransparency);
		iniWriteBool	(ini, sec, "restrictLightViewToScene",		mShadowMap.shadowMapUtil.restrictLightViewToScene);
		iniWriteFloat	(ini, sec, "bias",							mShadowMap.bias);
		iniWriteBool	(ini, sec, "autoCalcCascadeEnds",			mShadowMap.autoCalcCascadeEnds);
		iniWriteInt		(ini, sec, "numCascades",					mShadowMap.numCascades); // ATTN on load!
		for (int i = 0; i < SHADOWMAP_MAX_CASCADES; ++i) {
			iniWriteFloat	(ini, sec, "cascadeEnd_"			+ std::to_string(i), mShadowMap.shadowMapUtil.cascadeEnd[i]);
			iniWriteBool	(ini, sec, "depthBias.enable_"		+ std::to_string(i), mShadowMap.depthBias[i].enable);
			iniWriteFloat	(ini, sec, "depthBias.constant_"	+ std::to_string(i), mShadowMap.depthBias[i].constant);
			iniWriteFloat	(ini, sec, "depthBias.slope_"		+ std::to_string(i), mShadowMap.depthBias[i].slope);
			iniWriteFloat	(ini, sec, "depthBias.clamp_"		+ std::to_string(i), mShadowMap.depthBias[i].clamp);
		}

		sec = "PathEditor";
		iniWriteBool	(ini, sec, "mCameraSpline.draw",			mCameraSpline.draw);
		iniWriteInt		(ini, sec, "mCameraSpline.drawNpoints",		mCameraSpline.drawNpoints);

		sec = "Notes";
		iniWriteText	(ini, sec, "notes",							mNotepad);

		mAntiAliasing.writeSettingsToIni(ini);
		writeCamPathToIni(ini);

		return file.generate(ini);
	}

	bool loadSettings(std::string fullFilename) {
		mINI::INIFile file(fullFilename);
		mINI::INIStructure ini;
		std::string sec;

		if (!file.read(ini)) return false;
		
		sec = "Lights";
		iniReadInt		(ini, sec, "mMaxPointLightCount",			mMaxPointLightCount);
		iniReadInt		(ini, sec, "mMaxSpotLightCount",			mMaxSpotLightCount);
		iniReadVec3		(ini, sec, "mDirLight.intensity",			mDirLight.intensity);
		iniReadVec3		(ini, sec, "mDirLight.dir",					mDirLight.dir);
		iniReadFloat	(ini, sec, "mDirLight.boost",				mDirLight.boost);
		iniReadVec3		(ini, sec, "mAmbLight.col",					mAmbLight.col);
		iniReadFloat	(ini, sec, "mAmbLight.boost",				mAmbLight.boost);
		iniReadInt		(ini, sec, "mLightingMode",					mLightingMode);

		sec = "Rendering";
		iniReadFloat	(ini, sec, "mAlphaThreshold",				mAlphaThreshold);
		iniReadBool		(ini, sec, "mUseAlphaBlending",				mUseAlphaBlending);
		iniReadFloat	(ini, sec, "mLodBias",						mLodBias);
		iniReadBool		(ini, sec, "mLoadBiasTaaOnly",				mLoadBiasTaaOnly);
		iniReadBool		(ini, sec, "mAlwaysUseLod0",				mAlwaysUseLod0);
		iniReadFloat	(ini, sec, "mNormalMappingStrength",		mNormalMappingStrength);

		sec = "Camera";
		auto camT = mQuakeCam.translation();
		auto camR = mQuakeCam.rotation();
		iniReadVec3		(ini, sec, "mQuakeCam.translation",			camT); mQuakeCam.set_translation(camT);
		iniReadQuat		(ini, sec, "mQuakeCam.rotation",			camR); mQuakeCam.set_rotation(camR);
		iniReadBool		(ini, sec, "mAutoMovement",					mAutoMovement);
		iniReadInt		(ini, sec, "mAutoMovementUnits",			mAutoMovementUnits);
		iniReadBool		(ini, sec, "mAutoRotate",					mAutoRotate);
		iniReadVec2		(ini, sec, "mAutoRotateDegrees",			mAutoRotateDegrees);
		iniReadBool		(ini, sec, "mAutoBob",						mAutoBob);
		iniReadBool		(ini, sec, "mAutoStrafe",					mAutoStrafe);
		iniReadFloat	(ini, sec, "mAutoStrafeSpeed",				mAutoStrafeSpeed);
		iniReadFloat	(ini, sec, "mAutoStrafeDist",				mAutoStrafeDist);
		iniReadBool		(ini, sec, "mCameraSpline.enable",			mCameraSpline.enable);	// ATTN on load
		iniReadBool		(ini, sec, "mCameraSpline.cyclic",			mCameraSpline.cyclic);
		//iniReadBool	(ini, sec, "mCameraSpline.lookAlong",		mCameraSpline.lookAlong);	// this is stored with cam path

		sec = "DynObjects";
		iniReadBool		(ini, sec, "mMovingObject.enabled",			mMovingObject.enabled);
		iniReadInt		(ini, sec, "mMovingObject.moverId",			mMovingObject.moverId);
		iniReadVec3		(ini, sec, "mMovingObject.startPos",		mMovingObject.startPos);
		iniReadVec3		(ini, sec, "mMovingObject.endPos",			mMovingObject.endPos);
		iniReadVec4		(ini, sec, "mMovingObject.rotAxisAngle",	mMovingObject.rotAxisAngle);
		iniReadFloat	(ini, sec, "mMovingObject.speed",			mMovingObject.speed);
		iniReadInt		(ini, sec, "mMovingObject.units",			mMovingObject.units);
		iniReadInt		(ini, sec, "mMovingObject.repeat",			mMovingObject.repeat);
		iniReadBool		(ini, sec, "mMovingObject.rotContinous",	mMovingObject.rotContinous);
		iniReadFloat	(ini, sec, "dynObj.mAnimTime",				mDynObjects[mMovingObject.moverId].mAnimTime);
		iniReadBool		(ini, sec, "mMovingObject.autoAnimate",		mMovingObject.autoAnimate);
		iniReadFloat	(ini, sec, "mMovingObject.animSpeed",		mMovingObject.animSpeed);

		sec = "Images";
		iniReadBool		(ini, sec, "mTestImageSettings.enabled",	mTestImageSettings.enabled);
		iniReadInt		(ini, sec, "mTestImageSettings.imageId",	mTestImageSettings.imageId);
		iniReadBool		(ini, sec, "mTestImageSettings.bilinear",	mTestImageSettings.bilinear);

		sec = "Shadows";
		iniReadBool		(ini, sec, "enable",						mShadowMap.enable);
		iniReadBool		(ini, sec, "enableForTransparency",			mShadowMap.enableForTransparency);
		iniReadBool		(ini, sec, "restrictLightViewToScene",		mShadowMap.shadowMapUtil.restrictLightViewToScene);
		iniReadFloat	(ini, sec, "bias",							mShadowMap.bias);
		iniReadBool		(ini, sec, "autoCalcCascadeEnds",			mShadowMap.autoCalcCascadeEnds);
		iniReadInt		(ini, sec, "numCascades",					mShadowMap.desiredNumCascades); // ATTN on load! read to desiredNumCascades
		for (int i = 0; i < SHADOWMAP_MAX_CASCADES; ++i) {
			iniReadFloat	(ini, sec, "cascadeEnd_"			+ std::to_string(i), mShadowMap.shadowMapUtil.cascadeEnd[i]);
			iniReadBool		(ini, sec, "depthBias.enable_"		+ std::to_string(i), mShadowMap.depthBias[i].enable);
			iniReadFloat	(ini, sec, "depthBias.constant_"	+ std::to_string(i), mShadowMap.depthBias[i].constant);
			iniReadFloat	(ini, sec, "depthBias.slope_"		+ std::to_string(i), mShadowMap.depthBias[i].slope);
			iniReadFloat	(ini, sec, "depthBias.clamp_"		+ std::to_string(i), mShadowMap.depthBias[i].clamp);
		}

		sec = "PathEditor";
		iniReadBool		(ini, sec, "mCameraSpline.draw",			mCameraSpline.draw);
		iniReadInt		(ini, sec, "mCameraSpline.drawNpoints",		mCameraSpline.drawNpoints);

		sec = "Notes";
		iniReadText		(ini, sec, "notes",							mNotepad);

		mAntiAliasing.readSettingsFromIni(ini);
		readCamPathFromIni(ini);

		// TODO: check all stuff we need to reset
		invalidate_command_buffers();

		return true;
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
	std::vector<dynamic_object> mDynObjects;
	std::array<avk::command_buffer, cConcurrentFrames> mModelsCommandBuffer;
	std::array<avk::fence, cConcurrentFrames> mModelsCommandBufferFence;

	// shadowmap
	avk::renderpass mShadowmapRenderpass;
	avk::graphics_pipeline mPipelineShadowmapOpaque, mPipelineShadowmapTransparent, mPipelineShadowmapAnimObject, mPipelineDrawShadowmap, mPipelineDrawFrustum;
	std::array<avk::command_buffer, cConcurrentFrames> mShadowmapCommandBuffer;
	struct ShadowMapPerCascadeResources {
		std::array<avk::framebuffer, cConcurrentFrames> mShadowmapFramebuffer;
	};
	std::array<ShadowMapPerCascadeResources, SHADOWMAP_MAX_CASCADES> mShadowmapPerCascade;
	std::array<std::array<avk::image_sampler, SHADOWMAP_MAX_CASCADES>, cConcurrentFrames> mShadowmapImageSamplers;

	// GPU frustum culling
	avk::compute_pipeline mPipelineFrustumCulling, mPipelineBuildSceneBuffers;

	avk::sampler mGenericSamplerNearestNeighbour;

	// Different pipelines used for (deferred) shading:
	avk::graphics_pipeline mPipelineFirstPass;
	avk::graphics_pipeline mPipelineLightingPass;

	// Pipelines for forward rendering:
	avk::graphics_pipeline mPipelineFwdOpaque;
	avk::graphics_pipeline mPipelineFwdTransparent;
	avk::graphics_pipeline mPipelineFwdTransparentNoBlend;

	// Common pipelines:
	avk::graphics_pipeline mPipelineAnimObject;
	avk::graphics_pipeline mPipelineDrawCamPath;
	avk::graphics_pipeline mPipelineTestImage;

	const int mMaxCamPathPositions = 10'000;
	int mNumCamPathPositions = 0;
	bool mDrawCamPathPositions_valid = false;
	avk::buffer mDrawCamPathPositionsBuffer;

	// The elements to handle the post processing effects:
	taa<cConcurrentFrames> mAntiAliasing;

#if USE_GVK_UPDATER
	//gvk::updater mUpdater;	// handles shader hot reloading, window resizing
#endif

	int mLightingMode = 0; // 0 = typical; 1 = no lights, just diff color;  2 = debug
	struct { glm::vec3 dir, intensity; float boost; } mDirLight = { {1.f,1.f,1.f}, { 1.f,1.f,1.f }, 1.f }; // this is overwritten with the dir light from the .fscene file
	struct { glm::vec3 col; float boost; } mAmbLight = { {1.f, 1.f, 1.f}, 0.1f };

	float mAlphaThreshold = 0.5f; // 0.001f; // alpha threshold for rendering transparent parts (0.5 ok for alpha-testing, 0.001 for alpha-blending)
	float mLodBias;
	bool mLoadBiasTaaOnly = true;
	bool mAlwaysUseLod0 = false;

	glm::vec2 mAutoRotateDegrees = glm::vec2(-45, 0);
	bool mAutoRotate = false;
	bool mAutoBob = false;
	bool mAutoStrafe = false;
	float mAutoStrafeSpeed = 3.f;
	float mAutoStrafeDist = 0.5f;
	bool mAutoMovement = true;
	int mAutoMovementUnits = 0; // 0 = per sec, 1 = per frame

	int mMovingObjectFirstMatIdx = -1;

	bool mReRecordCommandBuffers = false;

	struct {
		bool      enabled = false;
		int       moverId = 0;
		glm::vec3 translation = {};
		float     rotationAngle;	// radians
		glm::vec3 startPos = glm::vec3(0,1,0); // glm::vec3(-5, 1, 0);
		glm::vec3 endPos   = glm::vec3(0,1,0); // glm::vec3(5,1,0);
		glm::vec4 rotAxisAngle = glm::vec4(0,0,1,-90);		// axis, angle(degr.)
		float     speed    = 5.f;
		float     t = 0.f;
		int       units = 0; // 0 = per sec, 1 = per frame
		int       repeat = 1; // 0 = no, 1 = cycle, 2 = ping-pong
		bool      rotContinous = false;
		bool      autoAnimate = true;
		float     animSpeed = 1.f;
	} mMovingObject;

	struct {
		bool	enabled = false;
		int		imageId = 0;
		bool	bilinear = true;
	} mTestImageSettings;
	std::vector<avk::image_view> mTestImages;
	avk::sampler mTestImageSampler_bilinear;
	avk::sampler mTestImageSampler_nearest;


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

		// other static buffers
		avk::buffer mAttributesBuffer;			// per (global) mesh
		avk::buffer mMeshgroupInfoBuffer;		// for GPU-frustum culling
		avk::buffer mCullingBoundingBoxBuffer;

		// dynamic buffers
		std::array<avk::buffer, cConcurrentFrames> mDrawnMeshgroupBuffer;			// per drawn meshgroup: info for one (drawn) meshgroup
		std::array<avk::buffer, cConcurrentFrames> mDrawnMeshAttribIndexBuffer;		// per drawn mesh:      info for one (drawn) mesh
		std::array<avk::buffer, cConcurrentFrames> mDrawCommandsBuffer;				// draw parameters (VkDrawIndexedIndirectCommand)
		std::array<avk::buffer, cConcurrentFrames> mDrawCountBuffer;				// draw count for opaque and transparent meshgroups (only 2 entries: [0]=opaque [1]=transparent)
		std::array<avk::buffer, cConcurrentFrames> mCullingVisibilityBuffer;		// for GPU-frustum culling
		std::array<avk::buffer, cConcurrentFrames> mCullingUniformsBuffer;
		std::array<avk::buffer, cConcurrentFrames> mMeshgroupsLayoutInfoBuffer;		// TODO - can't we just offset gl_DrawID for transparent parts?

		// temporary vectors, holding data to be uploaded to the GPU
		std::vector<uint32_t> mIndices;
		std::vector<glm::vec3> mPositions;
		std::vector<glm::vec2> mTexCoords;
		std::vector<glm::vec3> mNormals;
		std::vector<glm::vec3> mTangents;
		std::vector<glm::vec3> mBitangents;

		// the mesh groups
		std::vector<Meshgroup> mMeshgroups;
		uint32_t mMaxOpaqueMeshgroups;
		uint32_t mMaxTransparentMeshgroups;
		size_t   mNumTotalInstances;

		// full scene bounding box
		BoundingBox mBoundingBox;

#if ENABLE_RAYTRACING
		// raytracing-stuff
		std::vector<avk::bottom_level_acceleration_structure> mBLASs;					// bottom level acceleration structures (one per object, shared for each TLAS)
		std::array<avk::top_level_acceleration_structure, cConcurrentFrames> mTLASs;	// top level acceleration structures, one per frame
		std::vector<avk::geometry_instance> mGeometryInstances;							// geometry instances for the BLASs of the static geometry
		std::vector<glm::mat4> mDebugGeoInstTransforms;
#endif

		bool mRegeneratePerFrame = true;
		bool mCullViewFrustum = true;

		void print_stats() {
			size_t numIns[2] = { 0, 0 }, numGrp[2] = { 0, 0 };
			for (auto &mg : mMeshgroups) {
				numIns[mg.hasTransparency ? 1 : 0] += mg.perInstanceData.size();
				numGrp[mg.hasTransparency ? 1 : 0] ++;
			}
			printf("Scene stats:  groups:    opaque %5lld, transparent %5lld, total %5lld\n", numGrp[0], numGrp[1], numGrp[0] + numGrp[1]);
			printf("              instances: opaque %5lld, transparent %5lld, total %5lld\n", numIns[0], numIns[1], numIns[0] + numIns[1]);
			printf("Scene bounds: min %.2f %.2f %.2f,  max %.2f %.2f %.2f,  diag %.2f\n", mBoundingBox.min.x, mBoundingBox.min.y, mBoundingBox.min.z, mBoundingBox.max.x, mBoundingBox.max.y, mBoundingBox.max.z, glm::distance(mBoundingBox.min, mBoundingBox.max));
		}
	} mSceneData;

	struct {
		bool   enable;
		float  tStart;
		bool   cyclic = true;
		//Spline spline = Spline(8.f, { {-1,0,0},{0,0,0},{1,0,0},{1,0,10},{0,0,10},{0,0,0},{0,0,-1} });
		bool   draw = false;
		int    drawNpoints = 5000;
		bool   editorControlPointsValid = false;

		float duration = 8.f;
		bool  constantSpeed = false;
		bool  lookAlong = false;
		InterpolationCurve interpolationCurve = InterpolationCurve(InterpolationCurveType::catmull_rom_spline, { {-1,0,0},{0,0,0},{1,0,0},{1,0,10},{0,0,10},{0,0,0},{0,0,-1} });
	} mCameraSpline;

	glm::uvec2 mHiResolution, mLoResolution;

	std::array<avk::buffer, cConcurrentFrames> mBoneMatricesBuffer;
	std::array<avk::buffer, cConcurrentFrames> mBoneMatricesPrevBuffer;

	std::string mNotepad = "";

	struct {
		float bias = 0.f; //0.002f;
		int numCascades = SHADOWMAP_INITIAL_CASCADES;
		int desiredNumCascades = numCascades;
		bool autoCalcCascadeEnds = true;

		struct ShadowMapDepthBias {
			bool enable;
			float constant, clamp, slope;
		};
		ShadowMapDepthBias depthBias[SHADOWMAP_MAX_CASCADES] = {
			{ true, 0.0f, 0.0f, 1.0f},
			{ true, 0.0f, 0.0f, 1.0f},
			{ true, 0.0f, 0.0f, 1.0f},
			{ true, 0.0f, 0.0f, 1.0f},
		};

		bool enable = false; // true;
		bool enableForTransparency = false;// true;
		bool show = false;
		bool drawFrustum = false;
		ShadowMap shadowMapUtil;
	} mShadowMap;

	glm::mat4 effectiveCam_view_matrix() { return mEffectiveCamera.detached ? mEffectiveCamera.mViewMatrix  : mQuakeCam.view_matrix(); }
	glm::mat4 effectiveCam_proj_matrix() { return mEffectiveCamera.detached ? mEffectiveCamera.mProjMatrix  : mQuakeCam.projection_matrix(); }
	glm::vec3 effectiveCam_translation() { return mEffectiveCamera.detached ? mEffectiveCamera.mTranslation : mQuakeCam.translation(); }
	glm::quat effectiveCam_rotation()    { return mEffectiveCamera.detached ? mEffectiveCamera.mRotation    : mQuakeCam.rotation(); }
	struct {
		bool detached = false;
		glm::mat4 mViewMatrix;
		glm::mat4 mProjMatrix;
		glm::vec3 mTranslation;
		glm::quat mRotation;
		void storeValuesFrom(gvk::quake_camera &cam) {
			mViewMatrix  = cam.view_matrix();
			mProjMatrix  = cam.projection_matrix();
			mTranslation = cam.translation();
			mRotation    = cam.rotation();
		}
	} mEffectiveCamera;

#if ENABLE_RAYTRACING
	std::array<avk::image_view, cConcurrentFrames> mRtImageViews;					// target images for ray tracing
	avk::image_view mRtDummySegMask;
	avk::image_view mRtDummyInputImg;
	avk::ray_tracing_pipeline mPipelineRayTrace;
	std::vector<avk::buffer_view> mRtIndexBuffersArray;
	std::vector<avk::buffer_view> mRtTexCoordBuffersArray;
	std::vector<avk::buffer_view> mRtNormalsBuffersArray;
	std::vector<avk::buffer_view> mRtTangentsBuffersArray;
	std::vector<avk::buffer_view> mRtBitangentsBuffersArray;
	std::vector<avk::buffer_view> mRtPositionsBuffersArray;
	avk::buffer mRtMaterialIndexBuffer;
	avk::buffer mRtPixelOffsetBuffer;
	std::vector<avk::geometry_instance> mAllGeometryInstances;					// all geometry instances in the current TLAS
	std::array<avk::buffer, cConcurrentFrames> mRtAnimObjNormalsBuffer;			// one normals buffer for *all* meshes of the current animated object
	std::array<avk::buffer, cConcurrentFrames> mRtAnimObjTangentsBuffer;		// same for tangents
	std::array<avk::buffer, cConcurrentFrames> mRtAnimObjBitangentsBuffer;		// same for bitangents
	std::array<avk::buffer, cConcurrentFrames> mRtAnimObjPositionsBuffer;		// positions for Lod approximation
	std::array<avk::buffer, cConcurrentFrames> mRtAnimObjNTBOffsetBuffer;		// start index of first normal/tangent/bitangent in above buffer for each mesh of the current animated object
	std::array<avk::buffer_view, cConcurrentFrames> mRtAnimObjNormalsBufferView;		// for uniform_texel_buffer
	std::array<avk::buffer_view, cConcurrentFrames> mRtAnimObjTangentsBufferView;		// for uniform_texel_buffer
	std::array<avk::buffer_view, cConcurrentFrames> mRtAnimObjBitangentsBufferView;		// for uniform_texel_buffer
	std::array<avk::buffer_view, cConcurrentFrames> mRtAnimObjPositionsBufferView;		// for uniform_texel_buffer
#endif
	int mRtSamplesPerPixel = 4;
	bool mRtDebugSparse = false;
	bool mRtApproximateLod = false;

	SceneType mSceneType = SceneType::Unknown;
};

static std::filesystem::path getExecutablePath() {
	wchar_t path[MAX_PATH] = { 0 };
	GetModuleFileNameW(NULL, path, MAX_PATH);
	return path;
}

int main(int argc, char **argv) // <== Starting point ==
{
#if SET_WORKING_DIRECTORY
	// set working directory to directory of executable
	try {
		std::filesystem::current_path(getExecutablePath().parent_path());
	} catch (...) {
		printf("ERROR: Failed to set working directory!\n");
		// continue anyway...
	}
#endif

	// Init Renderdoc API
	rdoc::init();

	try {
		// Parse command line
		// first parameter starting without dash is scene filename
		// -- (double-dash) terminates command line, everything after (including scene file name) is ignored
		const glm::ivec2 win_size_def   = { 1920, 1080 };
		const glm::ivec2 win_size_small = { 1280,  720 };
		bool badCmd = false;
		bool disableValidation = false;
		bool forceValidation = false;
		bool disableMip = false;
		bool enableAlphaBlending = false;
		bool disableAlphaBlending = false;
		int  capture_n_frames = 0;
		bool skip_scene_filename = false;
		int window_width  = win_size_def.x;
		int window_height = win_size_def.y;
		bool hide_window = false;
		float upsample_factor = 1.f;
		bool fullscreen = false;
		bool vali_GpuAssisted = false;
		bool vali_BestPractices = false;
		std::string devicehint = "";

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
				} else if (0 == _stricmp(argv[i], "-gpuassist") || 0 == _stricmp(argv[i], "-gpuassisted")) {
					vali_GpuAssisted = true;
				} else if (0 == _stricmp(argv[i], "-bestpractice") || 0 == _stricmp(argv[i], "-bestpractices")) {
					vali_BestPractices = true;
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
					//if (capture_n_frames < 1) { badCmd = true; break; }
				} else if (0 == _stricmp(argv[i], "-hidewindow")) {
					hide_window = true;
				} else if (0 == _stricmp(argv[i], "-sponza")) {
					skip_scene_filename = true;
				} else if (0 == _stricmp(argv[i], "-test")) {
					skip_scene_filename = true;
					sceneFileName = "../../extras/TestScene/TestScene.fscene";
				} else if (0 == _stricmp(argv[i], "-small")) {
					window_width  = win_size_small.x;
					window_height = win_size_small.y;
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
				} else if (0 == _stricmp(argv[i], "-fullscreen") || 0 == _stricmp(argv[i], "-full")) {
					fullscreen = true;
				} else if (0 == _stricmp(argv[i], "-upsample") || 0 == _stricmp(argv[i], "-upsampling")) {
					i++;
					if (i >= argc) { badCmd = true; break; }
					upsample_factor = (float)atof(argv[i]);
				} else if (0 == _stricmp(argv[i], "-device")) {
					i++;
					if (i >= argc) { badCmd = true; break; }
					devicehint = argv[i];
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
			std::cout << "Usage: " << getExecutablePath().filename().string() << " [optional parameters] [orca scene file path]" << std::endl;
			//printf("Usage: %s [optional parameters] [orca scene file path]\n", argv[0]);
			std::cout << "Parameters:\n"
				"-w <width>             set window width\n"
				"-h <height>            set window height\n"
				"-small                 use smaller default window size (" << win_size_small.x << "x" << win_size_small.y << ") instead of (" << win_size_def.x << "x" << win_size_def.y << ")\n"
				"-fullscreen            enable fullscreen mode\n"
				"-upsample <factor>     upsampling factor (render framebuffer is <factor> times smaller than the window)\n"
				"-sponza                ignore scene file path and load Sponza scene\n"
				"-test                  ignore scene file path and load Test scene\n"
				"-device <hint>         device hint for GPU selection (e.g., -device INTEL or -device RTX)\n"
				"-novalidation          disable validation layers (in debug builds)\n"
				"-validation            enable validation layers (in release builds)\n"
				"-gpuassisted           enable GPU-Assisted validation extension\n"
				"-bestpractices         enable best practices validation extension\n"
				"-blend                 use alpha blending for transparent parts\n"
				"-noblend               use alpha testing for transparent parts\n"
				"-nomip                 disable mip-map generation for loaded textures\n"
				"-hidewindow            hide render window while scene loading is in progress\n"
				"-capture <numFrames>   capture the first <numFrames> with RenderDoc (only when started FROM RenderDoc)\n"
				"--                     terminate argument list, everything after is ignored\n"
				<< std::endl;
			return EXIT_FAILURE;
		}

		if (upsample_factor < 1.f) { printf("Upsampling factor must be >= 1\n"); return EXIT_FAILURE; }

		// Create a window and open it
		auto mainWnd = gvk::context().create_window("TAA-STAR");
		mainWnd->set_resolution({ window_width, window_height });
		mainWnd->set_additional_back_buffer_attachments({ 
			avk::attachment::declare(vk::Format::eD32Sfloat, avk::on_load::clear, avk::depth_stencil(), avk::on_store::dont_care)
		});
		mainWnd->set_presentaton_mode(gvk::presentation_mode::mailbox);
		mainWnd->set_number_of_presentable_images(wookiee::cSwapchainImages);
		mainWnd->set_number_of_concurrent_frames (wookiee::cConcurrentFrames);
		mainWnd->request_srgb_framebuffer(true);
		mainWnd->open();

		if (fullscreen) mainWnd->switch_to_fullscreen_mode();

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
		chewbacca.mUpsamplingFactor = upsample_factor;
		chewbacca.mUpsampling = upsample_factor > 1.f;

		// setup capturing if RenderDoc is active
		if (capture_n_frames != 0 && rdoc::active()) {
			chewbacca.mCaptureNumFrames = abs(capture_n_frames);
			chewbacca.mStartCapture = true;
			chewbacca.mStartCaptureEarly = capture_n_frames < 0;
		}

		auto modifyValidationFunc = [&](gvk::validation_layers &val_layers) {
			// ac: disable or enforce validation layers via command line (renderdoc crashes when they are enabled....)
			val_layers.enable_in_release_mode(forceValidation); // note: in release, this doesn't enable the debug callback, but val.errors are dumped to the console
			if (disableValidation) val_layers.mLayers.clear();

			bool haveVali = (IF_DEBUG_BUILD_ELSE(true,false) || forceValidation) && (!disableValidation);
			std::string vali_status = haveVali ? "basic" : "off";
			if (haveVali) {
				// enable some extra validation layers/features
				//val_layers.add_layer("VK_LAYER_LUNARG_assistant_layer"); // add the assistant layer
				if (vali_GpuAssisted) {
					val_layers.enable_feature(vk::ValidationFeatureEnableEXT::eGpuAssisted);
					LOG_INFO("GPU-assisted validation extension enabled via command line parameter.");
					vali_status += " + GpuAssisted";
				}
				if (vali_BestPractices) {
					val_layers.enable_feature(vk::ValidationFeatureEnableEXT::eBestPractices);	// too verbose for now
					LOG_INFO("Best-practices validation extension enabled via command line parameter.");
					vali_status += " + BestPractices";
				}
			}
			LOG_INFO("Final Validation layers: " + vali_status);
		};

		// request VK_EXT_debug_marker (ONLY if running from RenderDoc!) for object labeling
		gvk::required_device_extensions dev_extensions;
		for (auto ext : rdoc::required_device_extensions()) dev_extensions.add_extension(ext);

#if ENABLE_RAYTRACING
		// add extensions necessary for raytracing
		dev_extensions
			.add_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
			.add_extension(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME)
			.add_extension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)
			.add_extension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)
			.add_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
			.add_extension(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
#endif

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
			},
			[](vk::PhysicalDeviceVulkan12Features& pdf) {
				pdf.drawIndirectCount = VK_TRUE;	// needed for vkCmdDrawIndexedIndirectCount
#if ENABLE_RAYTRACING
				pdf.setBufferDeviceAddress(VK_TRUE);
#endif
			},
#if ENABLE_RAYTRACING
			[](vk::PhysicalDeviceRayTracingPipelineFeaturesKHR& rtpf) {
				rtpf.setRayTracingPipeline(VK_TRUE);
			},
			[](vk::PhysicalDeviceAccelerationStructureFeaturesKHR& asf) {
				asf.setAccelerationStructure(VK_TRUE);
			},
#endif
			gvk::physical_device_selection_hint(devicehint)
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
