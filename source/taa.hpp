#pragma once

#include <gvk.hpp>
#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include "helper_functions.hpp"

#include "debug_helper.hpp"

// This class handles the anti-aliasing post-processing effect(s).
// It is templated on the number of concurrent frames, i.e. some resources
// are created CF-times, once for each concurrent frame.
template <size_t CF>
class taa : public gvk::invokee
{
	struct push_constants_for_taa {
		glm::vec4 mJitterAndAlpha;
		int mColorClampingOrClipping;
		VkBool32 mDepthCulling;
		VkBool32 mTextureLookupUnjitter;
		VkBool32 mBypassHistoryUpdate;
		int mDebugMode;
		float mDebugScale;
	};

	struct matrices_for_taa {
		glm::mat4 mHistoryViewProjMatrix;
		glm::mat4 mInverseViewProjMatrix;
	};

public:
	taa(avk::queue* aQueue)
		: invokee("Temporal Anti-Aliasing Post Processing Effect")
		, mQueue{ aQueue }
	{ }

	// Execute after all previous post-processing effects:
	int execution_order() const override { return 100; }

	// ac: Halton sequence, see https://en.wikipedia.org/wiki/Halton_sequence
	static float halton(int i, int b) { // index, base; (index >= 1)
		float f = 1.0f, r = 0.0f;
		while (i > 0) {
			f = f / b;
			r = r + f * (i % b);
			i = i / b;
		}
		return r;
	}

	template <size_t Len>
	static std::array<glm::vec2, Len> halton_2_3(glm::vec2 aScale) {
		std::array<glm::vec2, Len> result;
		for (size_t i = 0; i < Len; i++) {
			result[i] = aScale * glm::vec2{ halton(int(i) + 1, 2) - 0.5f, halton(int(i) + 1, 3) - 0.5f };
		}
		return result;
	}
	
	// Compute an offset for the projection matrix based on the given frame-id
	glm::vec2 get_jitter_offset_for_frame(gvk::window::frame_id_t aFrameId) const
	{
		const static auto sResolution = gvk::context().main_window()->resolution();
		const static auto sPxSizeNDC = glm::vec2(2.0f / static_cast<float>(sResolution.x), 2.0f / static_cast<float>(sResolution.y));

		// Prepare some different distributions:
		const static auto sCircularQuadSampleOffsets = avk::make_array<glm::vec2>(
			sPxSizeNDC * glm::vec2(-0.25f, -0.25f),
			sPxSizeNDC * glm::vec2( 0.25f, -0.25f),
			sPxSizeNDC * glm::vec2( 0.25f,  0.25f),
			sPxSizeNDC * glm::vec2(-0.25f,  0.25f)
		);
		const static auto sUniform4HelixSampleOffsets = avk::make_array<glm::vec2>(
			sPxSizeNDC * glm::vec2(-0.25f, -0.25f),
			sPxSizeNDC * glm::vec2( 0.25f,  0.25f),
			sPxSizeNDC * glm::vec2( 0.25f, -0.25f),
			sPxSizeNDC * glm::vec2(-0.25f,  0.25f)
		);
		const static auto sHalton23x8SampleOffsets = halton_2_3<8>(sPxSizeNDC);
		const static auto sHalton23x16SampleOffsets = halton_2_3<16>(sPxSizeNDC);

		// Select a specific distribution:
		const glm::vec2* sampleOffsetValues = nullptr;
		size_t numSampleOffsets = 0;
		switch (mSampleDistribution) {
		case 0:
			sampleOffsetValues = sCircularQuadSampleOffsets.data();
			numSampleOffsets = sCircularQuadSampleOffsets.size();
			break;
		case 1:
			sampleOffsetValues = sUniform4HelixSampleOffsets.data();
			numSampleOffsets = sUniform4HelixSampleOffsets.size();
			break;
		case 2:
			sampleOffsetValues = sHalton23x8SampleOffsets.data();
			numSampleOffsets = sHalton23x8SampleOffsets.size();
			break;
		case 3:
			sampleOffsetValues = sHalton23x16SampleOffsets.data();
			numSampleOffsets = sHalton23x16SampleOffsets.size();
			break;
		}
		
		if (mJitterSlowMotion  > 1) aFrameId /= mJitterSlowMotion;
		if (mFixedJitterIndex >= 0) aFrameId = mFixedJitterIndex;


		auto pos = sampleOffsetValues[aFrameId % numSampleOffsets];

		if (mJitterRotateDegrees != 0.f) {
			float s = sin(glm::radians(mJitterRotateDegrees));
			float c = cos(glm::radians(mJitterRotateDegrees));
			pos = glm::vec2(pos.x * c - pos.y * s, pos.x * s + pos.y * c);
		}


		return pos * mJitterExtraScale;
	}

	void save_history_proj_matrix(glm::mat4 aProjMatrix, gvk::window::frame_id_t aFrameId)
	{
		assert(aFrameId == gvk::context().main_window()->current_frame());
		auto inFlightIndex = gvk::context().main_window()->in_flight_index_for_frame();
		mHistoryProjMatrices[inFlightIndex] = aProjMatrix;
	}
	
	// Applies a translation to the given matrix and returns the result
	glm::mat4 get_jittered_projection_matrix(glm::mat4 aProjMatrix, gvk::window::frame_id_t aFrameId) const
	{
		if (mTaaEnabled) {
			const auto xyOffset = get_jitter_offset_for_frame(aFrameId);
			return glm::translate(glm::vec3{xyOffset.x, xyOffset.y, 0.0f}) * aProjMatrix;
		}
		else {
			return aProjMatrix;
		}
	}

	// Store pointers to some resources passed from class wookiee::initialize(), 
	// and also create mHistoryImages, and mResultImages:
	template <typename SRCCOLOR, typename SRCD>
	void set_source_image_views(SRCCOLOR& aSourceColorImageViews, SRCD& aSourceDepthImageViews)
	{
		std::vector<avk::command_buffer> layoutTransitions;
		
		for (size_t i = 0; i < CF; ++i) {
			// Store pointers to the source color result images
			if constexpr (std::is_pointer<typename SRCCOLOR::value_type>::value) {
				mSrcColor[i] = &static_cast<avk::image_view_t&>(*aSourceColorImageViews[i]);
			} else {
				mSrcColor[i] = &static_cast<avk::image_view_t&>(aSourceColorImageViews[i]);
			}

			// Store pointers to the source depth images
			if constexpr (std::is_pointer<typename SRCD::value_type>::value) {
				mSrcDepth[i] = &static_cast<avk::image_view_t&>(*aSourceDepthImageViews[i]);
			}
			else {
				mSrcDepth[i] = &static_cast<avk::image_view_t&>(aSourceDepthImageViews[i]);
			}

			auto w = mSrcColor[i]->get_image().width();
			auto h = mSrcColor[i]->get_image().height();

			mResultImages[i] = gvk::context().create_image_view(
				gvk::context().create_image(w, h, vk::Format::eR8G8B8A8Unorm, 1, avk::memory_usage::device, avk::image_usage::general_storage_image)
			);
			rdoc::labelImage(mResultImages[i]->get_image().handle(), "taa.mResultImages", i);
			layoutTransitions.emplace_back(std::move(mResultImages[i]->get_image().transition_to_layout({}, avk::sync::with_barriers_by_return({}, {})).value()));

			mResultImagesSrgb[i] = gvk::context().create_image_view(
				gvk::context().create_image(w, h, vk::Format::eR8G8B8A8Srgb, 1, avk::memory_usage::device, avk::image_usage::general_image)
			);
			rdoc::labelImage(mResultImagesSrgb[i]->get_image().handle(), "taa.mResultImagesSrgb", i);
			layoutTransitions.emplace_back(std::move(mResultImagesSrgb[i]->get_image().transition_to_layout({}, avk::sync::with_barriers_by_return({}, {})).value()));

			mDebugImages[i] = gvk::context().create_image_view(
				gvk::context().create_image(w, h, vk::Format::eR8G8B8A8Unorm, 1, avk::memory_usage::device, avk::image_usage::general_storage_image)
			);
			rdoc::labelImage(mDebugImages[i]->get_image().handle(), "taa.mDebugImages", i);
			layoutTransitions.emplace_back(std::move(mDebugImages[i]->get_image().transition_to_layout({}, avk::sync::with_barriers_by_return({}, {})).value()));
		}

		std::vector<std::reference_wrapper<avk::command_buffer_t>> commandBufferReferences;
		std::transform(std::begin(layoutTransitions), std::end(layoutTransitions), std::back_inserter(commandBufferReferences), [](avk::command_buffer& cb) { return std::ref(*cb); });
		auto fen = mQueue->submit_with_fence(commandBufferReferences);
		fen->wait_until_signalled();
	}

	// Return a reference to all the result images:
	auto& result_images()
	{
		return mResultImagesSrgb;
	}

	// Return a reference to one particular result image:
	avk::image_view_t& result_image_at(size_t i)
	{
		return mResultImagesSrgb[i];
	}

	// Return the result of the GPU timer query:
	float duration()
	{
		if (!mTaaEnabled) {
			return 0.0f;
		}
		auto inFlightIndex = gvk::context().main_window()->in_flight_index_for_frame();
		return helpers::get_timing_interval_in_ms(fmt::format("TAA {}", inFlightIndex));
	}

	// Create all the compute pipelines used for the post processing effect(s),
	// prepare some command buffers with pipeline barriers to synchronize with subsequent commands,
	// create a new ImGui window that allows to enable/disable anti-aliasing, and to modify parameters:
	void initialize() override 
	{
		using namespace avk;
		using namespace gvk;
		
		// Create a descriptor cache that helps us to conveniently create descriptor sets:
		mDescriptorCache = gvk::context().create_descriptor_cache();
		
		mSampler = context().create_sampler(avk::filter_mode::bilinear, avk::border_handling_mode::clamp_to_border, 0);

		for (size_t i = 0; i < CF; ++i) {
			mMatricesBuffer[i] = context().create_buffer(memory_usage::host_coherent, {}, uniform_buffer_meta::create_from_size(sizeof(matrices_for_taa)));
		}

		mTaaPipeline = context().create_compute_pipeline_for(
			"shaders/taa.comp",
			descriptor_binding(0, 0, mSampler),
			descriptor_binding(0, 1, *mSrcColor[0]),
			descriptor_binding(0, 2, *mSrcDepth[0]),
			descriptor_binding(0, 3, *mResultImagesSrgb[0]),
			descriptor_binding(0, 4, *mSrcDepth[0]),
			descriptor_binding(0, 5, mResultImages[0]->as_storage_image()),
			descriptor_binding(0, 6, mDebugImages[0]->as_storage_image()),
			descriptor_binding(1, 0, mMatricesBuffer[0]),
			push_constant_binding_data { shader_type::compute, 0, sizeof(push_constants_for_taa) }
		);

		auto& commandPool = context().get_command_pool_for_reusable_command_buffers(*mQueue);
		
		// Record command buffers which contain pipeline barriers to properly synchronize with subsequent commands
		for (size_t i = 0; i < CF; ++i) {
			mSyncAfterCommandBuffers[i] = commandPool->alloc_command_buffer();
			mSyncAfterCommandBuffers[i]->begin_recording();
			mSyncAfterCommandBuffers[i]->establish_global_memory_barrier(
				// Sync between the following pipeline stages:
				pipeline_stage::compute_shader                        | pipeline_stage::transfer,             /* -> */ pipeline_stage::compute_shader                      | pipeline_stage::transfer,
				// According to those pipeline stages, sync the following memory accesses:
				memory_access::shader_buffers_and_images_write_access | memory_access::transfer_write_access, /* -> */ memory_access::shader_buffers_and_images_any_access | memory_access::transfer_read_access
			);
			mSyncAfterCommandBuffers[i]->end_recording();
		}
		
		// Create a new ImGui window:
		auto imguiManager = current_composition()->element_by_type<imgui_manager>();
		if(nullptr != imguiManager) {
			imguiManager->add_callback([this](){
				using namespace ImGui;

				Begin("Anti-Aliasing Settings");
				SetWindowPos(ImVec2(270.0f, 555.0f), ImGuiCond_FirstUseEver);
				SetWindowSize(ImVec2(220.0f, 130.0f), ImGuiCond_FirstUseEver);
				Checkbox("enabled", &mTaaEnabled);
				static const char* sColorClampingClippingValues[] = { "nope", "clamping", "clipping" };
				Combo("color clamping/clipping", &mColorClampingOrClipping, sColorClampingClippingValues, IM_ARRAYSIZE(sColorClampingClippingValues));
				Checkbox("depth culling", &mDepthCulling);
				Checkbox("texture lookup unjitter", &mTextureLookupUnjitter);
				static const char* sSampleDistributionValues[] = { "circular quad", "uniform4 helix", "halton(2,3) x8", "halton(2,3) x16" };
				Combo("sample distribution", &mSampleDistribution, sSampleDistributionValues, IM_ARRAYSIZE(sSampleDistributionValues));
				SliderFloat("alpha", &mAlpha, 0.0f, 1.0f);
				if (Button("reset")) mResetHistory = true;
				static const char* sImageToShowValues[] = { "result", "color bb (rgb)", "color bb(size)", "rejection" };
				Combo("display", &mImageToShow, sImageToShowValues, IM_ARRAYSIZE(sImageToShowValues));
				SliderFloat("scale##debug scale", &mDebugScale, 0.f, 20.f, "%.0f");

				if (CollapsingHeader("Jitter debug")) {
					SliderInt ("lock",		&mFixedJitterIndex, -1, 16);
					InputFloat("scale",		&mJitterExtraScale, 0.f, 0.f, "%.0f");
					InputInt  ("slowdown",	&mJitterSlowMotion);
					InputFloat("rotate",	&mJitterRotateDegrees);
				}
				End();
			});
		}
		else {
			LOG_WARNING("No component of type cgb::imgui_manager found => could not install ImGui callback.");
		}
	}

	// Update the push constant data that will be used in render():
	void update() override 
	{
		using namespace avk;
		using namespace gvk;
		
		auto inFlightIndex = context().main_window()->in_flight_index_for_frame();
		const auto* quakeCamera = current_composition()->element_by_type<quake_camera>();
		assert (nullptr != quakeCamera);

		// jitter-slow motion -> bypass history update on unchanged frames
		if (mJitterSlowMotion > 1) {
			static gvk::window::frame_id_t lastJitterIndex = 0;
			auto thisJitterIndex = gvk::context().main_window()->current_frame() / mJitterSlowMotion;
			mBypassHistoryUpdate = (thisJitterIndex == lastJitterIndex);
			lastJitterIndex = thisJitterIndex;
		} else {
			mBypassHistoryUpdate = false;
		}

		float effectiveAlpha = mResetHistory ? 1.f : mAlpha;
		mResetHistory = false;

		mHistoryViewMatrices[inFlightIndex] = quakeCamera->view_matrix();
		const auto jitter = get_jitter_offset_for_frame(gvk::context().main_window()->current_frame());
		mTaaPushConstants.mJitterAndAlpha = glm::vec4(jitter.x, jitter.y, 0.0f, effectiveAlpha);
		mTaaPushConstants.mColorClampingOrClipping = mColorClampingOrClipping;
		mTaaPushConstants.mDepthCulling = mDepthCulling;
		mTaaPushConstants.mTextureLookupUnjitter = mTextureLookupUnjitter;
		mTaaPushConstants.mBypassHistoryUpdate = mBypassHistoryUpdate;
		mTaaPushConstants.mDebugMode = mImageToShow;
		mTaaPushConstants.mDebugScale = mDebugScale;

	}

	// Create a new command buffer every frame, record instructions into it, and submit it to the graphics queue:
	void render() override
	{
		using namespace avk;
		using namespace gvk;
		
		auto* mainWnd = context().main_window();
		auto inFlightIndex = mainWnd->in_flight_index_for_frame();
		auto inFlightLastIndex = (inFlightIndex + CF - 1) % CF;

		auto& commandPool = context().get_command_pool_for_single_use_command_buffers(*mQueue);
		auto cmdbfr = commandPool->alloc_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
		cmdbfr->begin_recording();

		// ---------------------- If Anti-Aliasing is enabled perform the following actions --------------------------
		if (mTaaEnabled) {

			// fill matrices UBO
			matrices_for_taa matrices;
			matrices.mInverseViewProjMatrix = glm::inverse(mHistoryProjMatrices[inFlightIndex] * mHistoryViewMatrices[inFlightIndex]);
			matrices.mHistoryViewProjMatrix = mHistoryProjMatrices[inFlightLastIndex] * mHistoryViewMatrices[inFlightLastIndex];
			mMatricesBuffer[inFlightIndex]->fill(&matrices, 0, sync::not_required()); // sync is done with establish_global_memory_barrier below

			helpers::record_timing_interval_start(cmdbfr->handle(), fmt::format("TAA {}", inFlightIndex));

			cmdbfr->establish_global_memory_barrier(
				pipeline_stage::transfer,             /* -> */ pipeline_stage::compute_shader,
				memory_access::transfer_write_access, /* -> */ memory_access::shader_buffers_and_images_read_access
			);

			// We are going to use the previous in flight frame as history for TAA
			static_assert(CF > 1);

			// Apply Temporal Anti-Aliasing:
			cmdbfr->bind_pipeline(mTaaPipeline);
			cmdbfr->bind_descriptors(mTaaPipeline->layout(), mDescriptorCache.get_or_create_descriptor_sets({ 
				descriptor_binding(0, 0, mSampler),
				descriptor_binding(0, 1, *mSrcColor[inFlightIndex]),
				descriptor_binding(0, 2, *mSrcDepth[inFlightIndex]),
				descriptor_binding(0, 3, *mResultImagesSrgb[inFlightLastIndex]),
				descriptor_binding(0, 4, *mSrcDepth[inFlightLastIndex]),
				descriptor_binding(0, 5, mResultImages[inFlightIndex]->as_storage_image()),
				descriptor_binding(0, 6, mDebugImages[inFlightIndex]->as_storage_image()),
				descriptor_binding(1, 0, mMatricesBuffer[inFlightIndex])
			}));
			cmdbfr->push_constants(mTaaPipeline->layout(), mTaaPushConstants);
			cmdbfr->handle().dispatch((mResultImages[inFlightIndex]->get_image().width() + 15u) / 16u, (mResultImages[inFlightIndex]->get_image().height() + 15u) / 16u, 1);

			cmdbfr->establish_global_memory_barrier(
				pipeline_stage::compute_shader,                        /* -> */ pipeline_stage::transfer,
				memory_access::shader_buffers_and_images_write_access, /* -> */ memory_access::transfer_read_access
			);

			// Finally, copy into sRGB image:
			copy_image_to_another(mResultImages[inFlightIndex]->get_image(), mResultImagesSrgb[inFlightIndex]->get_image(), sync::with_barriers_into_existing_command_buffer(cmdbfr));
			cmdbfr->establish_global_memory_barrier(
				pipeline_stage::transfer,             /* -> */ pipeline_stage::transfer,
				memory_access::transfer_write_access, /* -> */ memory_access::transfer_read_access
			);
			// Blit into backbuffer directly from here (ATTENTION if you'd like to render something in other invokees!)
			//blit_image(mResultImagesSrgb[inFlightIndex]->get_image(), mainWnd->backbuffer_at_index(inFlightIndex).image_view_at(0)->get_image(), sync::with_barriers_into_existing_command_buffer(cmdbfr));

			auto &image_to_show = mImageToShow == 0 ? mResultImagesSrgb[inFlightIndex]->get_image() : mDebugImages[inFlightIndex]->get_image();
			blit_image(image_to_show, mainWnd->backbuffer_at_index(inFlightIndex).image_view_at(0)->get_image(), sync::with_barriers_into_existing_command_buffer(cmdbfr));

			helpers::record_timing_interval_end(cmdbfr->handle(), fmt::format("TAA {}", inFlightIndex));
		}
		// -------------------------- If Anti-Aliasing is disabled, do nothing but blit/copy ------------------------------
		else { 
			// Blit into backbuffer directly from here (ATTENTION if you'd like to render something in other invokees!)
			blit_image(mSrcColor[inFlightIndex]->get_image(), mResultImages[inFlightIndex]->get_image(),                                 sync::with_barriers_into_existing_command_buffer(cmdbfr));
			blit_image(mSrcColor[inFlightIndex]->get_image(), mResultImagesSrgb[inFlightIndex]->get_image(),                             sync::with_barriers_into_existing_command_buffer(cmdbfr));
			blit_image(mSrcColor[inFlightIndex]->get_image(), mainWnd->backbuffer_at_index(inFlightIndex).image_view_at(0)->get_image(), sync::with_barriers_into_existing_command_buffer(cmdbfr));
		}
		
		cmdbfr->end_recording();

		// The swap chain provides us with an "image available semaphore" for the current frame.
		// Only after the swapchain image has become available, we may start rendering into it.
		auto& imageAvailableSemaphore = mainWnd->consume_current_image_available_semaphore();
		
		// Submit the draw call and take care of the command buffer's lifetime:
		mQueue->submit(cmdbfr, imageAvailableSemaphore);
		mainWnd->handle_lifetime(std::move(cmdbfr));
	}

	void finalize() override
	{
	}

private:
	avk::queue* mQueue;
	avk::descriptor_cache mDescriptorCache;
	
	// Settings, which can be modified via ImGui:
	bool mTaaEnabled = true;
	int mColorClampingOrClipping = 1;
	bool mDepthCulling = false;
	bool mTextureLookupUnjitter = false;
	int mSampleDistribution = 0;
	float mAlpha = 0.1f;
	bool mResetHistory = false;

	// Source color images per frame in flight:
	std::array<avk::image_view_t*, CF> mSrcColor;
	std::array<avk::image_view_t*, CF> mSrcDepth;
	// Destination images per frame in flight:
	std::array<avk::image_view, CF> mResultImages;
	// Copying the result images into actually the same result images (but only sRGB format)
	// is a rather stupid workaround. It is also quite resource-intensive... Life's hard!
	std::array<avk::image_view, CF> mResultImagesSrgb;
	std::array<avk::image_view, CF> mDebugImages;
	// For each history frame's image content, also store the associated projection matrix:
	std::array<glm::mat4, CF> mHistoryProjMatrices;
	std::array<glm::mat4, CF> mHistoryViewMatrices;
	std::array<avk::buffer, CF> mMatricesBuffer;

	avk::sampler mSampler;
	
	// Prepared command buffers to synchronize subsequent commands
	std::array<avk::command_buffer, CF> mSyncAfterCommandBuffers;
	
	avk::compute_pipeline mTaaPipeline;
	push_constants_for_taa mTaaPushConstants;

	// jitter debugging
	int mFixedJitterIndex = -1;
	float mJitterExtraScale = 1.0f;
	int mJitterSlowMotion = 1;
	float mJitterRotateDegrees = 0.f;
	bool mBypassHistoryUpdate = false; // used by slow motion

	int mImageToShow = 0; // 0=result, 1=color bb (rgb), 2=color bb(size), 3=history rejection
	float mDebugScale = 1.f;
};
