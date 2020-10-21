#pragma once

#include <gvk.hpp>
#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include "imgui_helper.hpp"
#include "helper_functions.hpp"

#include "debug_helper.hpp"
#include "shader_cpu_common.h"

#define TAA_USE_POSTPROCESS_STEP 1

#if TAA_OUTPUT_IS_SRGB
#define RESULT_IMAGES_MAYBESRGB mResultImagesSrgb
#else
#define RESULT_IMAGES_MAYBESRGB mResultImages
#endif

// This class handles the anti-aliasing post-processing effect(s).
// It is templated on the number of concurrent frames, i.e. some resources
// are created CF-times, once for each concurrent frame.
template <size_t CF>
class taa : public gvk::invokee
{
	struct push_constants_for_taa {		// Note to self: be careful about alignment, esp. with vec#!
		glm::vec4 mJitterAndAlpha;
		int mColorClampingOrClipping;
		VkBool32 mDepthCulling;
		VkBool32 mTextureLookupUnjitter;
		VkBool32 mBypassHistoryUpdate;
		VkBool32 mUseYCoCg;
		VkBool32 mVarianceClipping;
		VkBool32 mLumaWeighting;
		float mVarClipGamma;
		float mMinAlpha;			// used for luminance-based weighting
		float mMaxAlpha;			// used for luminance-based weighting
		float mRejectionAlpha;
		VkBool32 mRejectOutside;
		int mUseVelocityVectors;		// 0=off 1=for movers only 2=for everything

		int mDebugMode;
		float mDebugScale;
		VkBool32 mDebugCenter;
	};

	struct push_constants_for_postprocess {	// !ATTN to alignment!
		glm::ivec4 zoomSrcLTWH	= { 960 - 10, 540 - 10, 20, 20 };
		glm::ivec4 zoomDstLTWH	= { 1920 - 200 - 10, 10, 200, 200 };
		VkBool32 zoom			= false;
		VkBool32 showZoomBox	= true;
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

	bool taa_enabled() const { return mTaaEnabled; }

	bool trigger_capture() {
		bool res = mTriggerCapture;
		mTriggerCapture = false;
		return res;
	}

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
			sPxSizeNDC * glm::vec2(0.25f, -0.25f),
			sPxSizeNDC * glm::vec2(0.25f, 0.25f),
			sPxSizeNDC * glm::vec2(-0.25f, 0.25f)
			);
		const static auto sUniform4HelixSampleOffsets = avk::make_array<glm::vec2>(
			sPxSizeNDC * glm::vec2(-0.25f, -0.25f),
			sPxSizeNDC * glm::vec2(0.25f, 0.25f),
			sPxSizeNDC * glm::vec2(0.25f, -0.25f),
			sPxSizeNDC * glm::vec2(-0.25f, 0.25f)
			);
		const static auto sHalton23x8SampleOffsets = halton_2_3<8>(sPxSizeNDC);
		const static auto sHalton23x16SampleOffsets = halton_2_3<16>(sPxSizeNDC);

		const static auto sDebugSampleOffsets = avk::make_array<glm::vec2>(sPxSizeNDC * glm::vec2(0, 0));

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
		case 4:
			sampleOffsetValues = sDebugSampleOffsets.data();
			numSampleOffsets = sDebugSampleOffsets.size();
			break;
		}

		if (mJitterSlowMotion > 1) aFrameId /= mJitterSlowMotion;
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
	glm::mat4 get_jittered_projection_matrix(glm::mat4 aProjMatrix, glm::vec2 &out_xyOffset, gvk::window::frame_id_t aFrameId) const
	{
		if (mTaaEnabled) {
			const auto xyOffset = get_jitter_offset_for_frame(aFrameId);
			out_xyOffset = xyOffset;
			return glm::translate(glm::vec3{ xyOffset.x, xyOffset.y, 0.0f }) * aProjMatrix;
		} else {
			out_xyOffset = glm::vec2(0);
			return aProjMatrix;
		}
	}

	glm::mat4 get_jittered_projection_matrix(glm::mat4 aProjMatrix, gvk::window::frame_id_t aFrameId) const
	{
		glm::vec2 dummy;
		return get_jittered_projection_matrix(aProjMatrix, dummy, aFrameId);
	}

	// Store pointers to some resources passed from class wookiee::initialize(), 
	// and also create mHistoryImages, and mResultImages:
	template <typename SRCCOLOR, typename SRCD, typename SRCVELOCITY>
	void set_source_image_views(SRCCOLOR& aSourceColorImageViews, SRCD& aSourceDepthImageViews, SRCVELOCITY& aSourceVelocityImageViews)
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
			} else {
				mSrcDepth[i] = &static_cast<avk::image_view_t&>(aSourceDepthImageViews[i]);
			}

			// Store pointers to the source velocity result images
			if constexpr (std::is_pointer<typename SRCVELOCITY::value_type>::value) {
				mSrcVelocity[i] = &static_cast<avk::image_view_t&>(*aSourceVelocityImageViews[i]);
			} else {
				mSrcVelocity[i] = &static_cast<avk::image_view_t&>(aSourceVelocityImageViews[i]);
			}


			auto w = mSrcColor[i]->get_image().width();
			auto h = mSrcColor[i]->get_image().height();

			mResultImages[i] = gvk::context().create_image_view(
				gvk::context().create_image(w, h, TAA_IMAGE_FORMAT_RGB, 1, avk::memory_usage::device, avk::image_usage::general_storage_image)
			);
			rdoc::labelImage(mResultImages[i]->get_image().handle(), "taa.mResultImages", i);
			layoutTransitions.emplace_back(std::move(mResultImages[i]->get_image().transition_to_layout({}, avk::sync::with_barriers_by_return({}, {})).value()));

#if TAA_OUTPUT_IS_SRGB
			mResultImagesSrgb[i] = gvk::context().create_image_view(
				//gvk::context().create_image(w, h, vk::Format::eR8G8B8A8Unorm, 1, avk::memory_usage::device, avk::image_usage::general_image)
				gvk::context().create_image(w, h, TAA_OUTPUT_IS_SRGB ? TAA_IMAGE_FORMAT_SRGB : TAA_IMAGE_FORMAT_RGB, 1, avk::memory_usage::device, avk::image_usage::general_image)
			);
			rdoc::labelImage(mResultImagesSrgb[i]->get_image().handle(), "taa.mResultImagesSrgb", i);
			layoutTransitions.emplace_back(std::move(mResultImagesSrgb[i]->get_image().transition_to_layout({}, avk::sync::with_barriers_by_return({}, {})).value()));
#endif

			mDebugImages[i] = gvk::context().create_image_view(
				gvk::context().create_image(w, h, vk::Format::eR16G16B16A16Sfloat, 1, avk::memory_usage::device, avk::image_usage::general_storage_image)
			);
			rdoc::labelImage(mDebugImages[i]->get_image().handle(), "taa.mDebugImages", i);
			layoutTransitions.emplace_back(std::move(mDebugImages[i]->get_image().transition_to_layout({}, avk::sync::with_barriers_by_return({}, {})).value()));

#if TAA_USE_POSTPROCESS_STEP
			mPostProcessImages[i] = gvk::context().create_image_view(
				gvk::context().create_image(w, h, TAA_IMAGE_FORMAT_POSTPROCESS, 1, avk::memory_usage::device, avk::image_usage::general_storage_image)
			);
			rdoc::labelImage(mPostProcessImages[i]->get_image().handle(), "taa.mPostProcessImages", i);
			layoutTransitions.emplace_back(std::move(mPostProcessImages[i]->get_image().transition_to_layout({}, avk::sync::with_barriers_by_return({}, {})).value()));
#endif
		}

		std::vector<std::reference_wrapper<avk::command_buffer_t>> commandBufferReferences;
		std::transform(std::begin(layoutTransitions), std::end(layoutTransitions), std::back_inserter(commandBufferReferences), [](avk::command_buffer& cb) { return std::ref(*cb); });
		auto fen = mQueue->submit_with_fence(commandBufferReferences);
		fen->wait_until_signalled();
	}

	//// Return a reference to all the result images:
	//auto& result_images()
	//{
	//	return mResultImagesSrgb;
	//}

	//// Return a reference to one particular result image:
	//avk::image_view_t& result_image_at(size_t i)
	//{
	//	return mResultImagesSrgb[i];
	//}

	// Return the result of the GPU timer query:
	float duration()
	{
		if (!mTaaEnabled) {
			return 0.0f;
		}
		auto inFlightIndex = gvk::context().main_window()->in_flight_index_for_frame();
		return helpers::get_timing_interval_in_ms(fmt::format("TAA {}", inFlightIndex));
	}


	void setup_ui_callback()
	{
		// Create a new ImGui window

		using namespace avk;
		using namespace gvk;

		auto imguiManager = current_composition()->element_by_type<imgui_manager>();
		if (nullptr != imguiManager) {
			imguiManager->add_callback([this]() {
				using namespace ImGui;
				using namespace imgui_helper;

				Begin("Anti-Aliasing Settings");
				SetWindowPos(ImVec2(270.0f, 555.0f), ImGuiCond_FirstUseEver);
				SetWindowSize(ImVec2(220.0f, 130.0f), ImGuiCond_FirstUseEver);
				Checkbox("enabled", &mTaaEnabled);
				SameLine(); if (Button("En&cap")) { mTaaEnabled = true; mTriggerCapture = true; }
				static const char* sColorClampingClippingValues[] = { "nope", "clamp", "clip fast", "clip slow" };
				Combo("color clamp/clip", &mColorClampingOrClipping, sColorClampingClippingValues, IM_ARRAYSIZE(sColorClampingClippingValues));
				Checkbox("variance clipping", &mVarianceClipping);
				SliderFloat("gamma", &mVarClipGamma, 0.f, 2.f, "%.2f");
				Checkbox("use YCoCg", &mUseYCoCg);
				Checkbox("luma weighting", &mLumaWeighting); HelpMarker("Set min and max alpha to define feedback range.");
				Checkbox("depth culling", &mDepthCulling);
				Checkbox("reject out-of-screen", &mRejectOutside);
				Checkbox("texture lookup unjitter", &mTextureLookupUnjitter);
				static const char* sSampleDistributionValues[] = { "circular quad", "uniform4 helix", "halton(2,3) x8", "halton(2,3) x16", "debug" };
				Combo("sample distribution", &mSampleDistribution, sSampleDistributionValues, IM_ARRAYSIZE(sSampleDistributionValues));
				SliderFloat("alpha", &mAlpha, 0.0f, 1.0f);
				SliderFloat("a_min", &mMinAlpha, 0.0f, 1.0f); HelpMarker("Luma weighting min alpha");
				SliderFloat("a_max", &mMaxAlpha, 0.0f, 1.0f); HelpMarker("Luma weighting max alpha");
				SliderFloat("rejection alpha", &mRejectionAlpha, 0.0f, 1.0f);
				Combo("use velocity", &mUseVelocityVectors, "none\0movers\0all\0");
				if (Button("reset")) mResetHistory = true;
				static const char* sDebugModeValues[] = { "color bb (rgb)", "color bb(size)", "rejection", "alpha", "velocity", "debug" /* always last */ };
				Checkbox("debug##show debug", &mShowDebug);
				SameLine();
				Combo("##debug mode", &mDebugMode, sDebugModeValues, IM_ARRAYSIZE(sDebugModeValues));
				PushItemWidth(100);
				SliderFloat("scale##debug scale", &mDebugScale, 0.f, 100.f, "%.0f");
				PopItemWidth();
				SameLine();
				Checkbox("center##debug center", &mDebugCenter);

				if (CollapsingHeader("Jitter debug")) {
					SliderInt("lock", &mFixedJitterIndex, -1, 16);
					InputFloat("scale", &mJitterExtraScale, 0.f, 0.f, "%.2f");
					InputInt("slowdown", &mJitterSlowMotion);
					InputFloat("rotate", &mJitterRotateDegrees);
				}

#if TAA_USE_POSTPROCESS_STEP
				if (CollapsingHeader("Postprocess")) {
					auto &pp = mPostProcessPushConstants;
					static bool pp_zoom = pp.zoom;					// ugly workaround for bool <-> VkBool32 conversion
					static bool pp_showZoomBox = pp.showZoomBox;

					static glm::ivec4 orig_zoomsrc = pp.zoomSrcLTWH;
					static glm::ivec4 orig_zoomdst = pp.zoomDstLTWH;

					Checkbox("enable", &mPostProcessEnabled);
					Checkbox("zoom", &pp_zoom);	
					SameLine(); Checkbox("show box", &pp_showZoomBox);
					SameLine(); if (Button("rst")) { pp.zoomSrcLTWH = orig_zoomsrc; pp.zoomDstLTWH = orig_zoomdst; }
					InputInt4("src", &pp.zoomSrcLTWH.x); HelpMarker("Left/Top/Width/Height");
					InputInt4("dst", &pp.zoomDstLTWH.x);

					pp.zoom = pp_zoom;
					pp.showZoomBox = pp_showZoomBox;
				}
#endif

				End();
				});
		} else {
			LOG_WARNING("No component of type cgb::imgui_manager found => could not install ImGui callback.");
		}
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
			descriptor_binding(0, 3, *RESULT_IMAGES_MAYBESRGB[0]),
			descriptor_binding(0, 4, *mSrcDepth[0]),
			descriptor_binding(0, 5, mResultImages[0]->as_storage_image()),
			descriptor_binding(0, 6, mDebugImages[0]->as_storage_image()),
			descriptor_binding(0, 7, *mSrcVelocity[0]),
			descriptor_binding(1, 0, mMatricesBuffer[0]),
			push_constant_binding_data{ shader_type::compute, 0, sizeof(push_constants_for_taa) }
		);

#if TAA_USE_POSTPROCESS_STEP
		mPostProcessPipeline = context().create_compute_pipeline_for(
			"shaders/post_process.comp",
			descriptor_binding(0, 1, *mResultImages[0]),
			descriptor_binding(0, 2, mPostProcessImages[0]->as_storage_image()),
			push_constant_binding_data{ shader_type::compute, 0, sizeof(push_constants_for_postprocess) }
		);
#endif


		auto& commandPool = context().get_command_pool_for_reusable_command_buffers(*mQueue);

		// Record command buffers which contain pipeline barriers to properly synchronize with subsequent commands
		for (size_t i = 0; i < CF; ++i) {
			mSyncAfterCommandBuffers[i] = commandPool->alloc_command_buffer();
			mSyncAfterCommandBuffers[i]->begin_recording();
			rdoc::beginSection(mSyncAfterCommandBuffers[i]->handle(), "TAA Sync", i);
			mSyncAfterCommandBuffers[i]->establish_global_memory_barrier(
				// Sync between the following pipeline stages:
				pipeline_stage::compute_shader | pipeline_stage::transfer,             /* -> */ pipeline_stage::compute_shader | pipeline_stage::transfer,
				// According to those pipeline stages, sync the following memory accesses:
				memory_access::shader_buffers_and_images_write_access | memory_access::transfer_write_access, /* -> */ memory_access::shader_buffers_and_images_any_access | memory_access::transfer_read_access
			);
			rdoc::endSection(mSyncAfterCommandBuffers[i]->handle());
			mSyncAfterCommandBuffers[i]->end_recording();
		}

		setup_ui_callback();
	}

	enum struct boxHitTestResult {
		left, top, right, bottom,
		left_top, right_top, left_bottom, right_bottom,
		inside, outside
	};
	boxHitTestResult boxHitTest(glm::ivec2 p, glm::ivec4 boxLTWH, bool checkCorners) {
		//return p.x >= boxLTWH.x && p.x < boxLTWH.x + boxLTWH.z && p.y >= boxLTWH.y && p.y < boxLTWH.y + boxLTWH.w;
		const int tolerance = 2;

		int dL = p.x - boxLTWH.x;
		int dR = boxLTWH.x + boxLTWH.z - 1 - p.x;
		int dT = p.y - boxLTWH.y;
		int dB = boxLTWH.y + boxLTWH.w - 1 - p.y;

		bool l = abs(dL) <= tolerance;
		bool r = abs(dR) <= tolerance;
		bool t = abs(dT) <= tolerance;
		bool b = abs(dB) <= tolerance;

		if (checkCorners) {
			if      (l && t) return boxHitTestResult::left_top;
			else if (r && t) return boxHitTestResult::right_top;
			else if (l && b) return boxHitTestResult::left_bottom;
			else if (r && b) return boxHitTestResult::right_bottom;
		}
		if      (l)      return boxHitTestResult::left;
		else if (r)      return boxHitTestResult::right;
		else if (t)      return boxHitTestResult::top;
		else if (b)      return boxHitTestResult::bottom;
		else if (dL >= 0 && dR >= 0 && dT >= 0 && dB >= 0) return boxHitTestResult::inside;
		else             return boxHitTestResult::outside;
	}

	// handle user input
	void handle_input() {
		using namespace gvk;
		const auto* quakeCamera = current_composition()->element_by_type<quake_camera>();
		if (quakeCamera && quakeCamera->is_enabled()) return;

		ImGuiIO &io = ImGui::GetIO();
		if (io.WantCaptureMouse) return;

		if (!mPostProcessPushConstants.zoom) return;

		static glm::ivec2 prev_pos = {0,0};
		static bool prev_lmb = false;
		static boxHitTestResult prev_ht = boxHitTestResult::outside;
		static int prev_box = -1;

		static std::vector<glm::ivec4 *> boxes = { &mPostProcessPushConstants.zoomSrcLTWH, &mPostProcessPushConstants.zoomDstLTWH };

		static glm::ivec2 drag_start_pos = {0,0};

		glm::ivec2 pos = glm::ivec2(input().cursor_position());
		bool lmb = (input().mouse_button_down(GLFW_MOUSE_BUTTON_LEFT));
		bool click = lmb && !prev_lmb;
		bool drag  = lmb && prev_lmb;
		
		// when dragging, keep previous hit-test
		auto ht = prev_ht;
		int box = prev_box;
		if (!drag) {

			ht = boxHitTestResult::outside; box = -1;
			for (auto i = 0; i < boxes.size(); ++i) {
				auto ht_i = boxHitTest(pos, *(boxes[i]), true);
				if (ht_i != boxHitTestResult::outside) {
					ht = ht_i; box = i; break;
				}
			}

			cursor cur = cursor::arrow_cursor;
			switch (ht) {
			case boxHitTestResult::left:
			case boxHitTestResult::right:			context().main_window()->set_cursor_mode(cursor::horizontal_resize_cursor);	break;
			case boxHitTestResult::top:
			case boxHitTestResult::bottom:			context().main_window()->set_cursor_mode(cursor::vertical_resize_cursor);	break;
			case boxHitTestResult::left_top:
			case boxHitTestResult::right_bottom:	context().main_window()->set_cursor_mode(cursor::crosshair_cursor);			break; // nw_or_se_resize_cursor	// diagonal cursors don't work with current glfw version :(
			case boxHitTestResult::right_top:
			case boxHitTestResult::left_bottom:		context().main_window()->set_cursor_mode(cursor::crosshair_cursor);			break; // ne_or_sw_resize_cursor
			case boxHitTestResult::inside:			context().main_window()->set_cursor_mode(cursor::hand_cursor);				break;
			default:								context().main_window()->set_cursor_mode(cursor::arrow_cursor);				break;
			}
		}

		if (lmb && (box >= 0)) {
			auto &pc = mPostProcessPushConstants;
			glm::ivec2 dpos = pos - prev_pos;
			glm::ivec4 newLTWH = *(boxes[box]);
			glm::ivec4 oldLTWH = newLTWH;
			if (click) {
				drag_start_pos = pos;
				if (ht == boxHitTestResult::outside) {
					newLTWH.x = pos.x - newLTWH.z / 2;
					newLTWH.y = pos.y - newLTWH.w / 2;
				}
			} else if (drag) {
				if (ht == boxHitTestResult::inside) {
					newLTWH.x += dpos.x;
					newLTWH.y += dpos.y;
				}
				if (ht == boxHitTestResult::left || ht == boxHitTestResult::left_top || ht == boxHitTestResult::left_bottom) {
					newLTWH.x += dpos.x;
					newLTWH.z -= dpos.x;
				}
				if (ht == boxHitTestResult::right || ht == boxHitTestResult::right_top || ht == boxHitTestResult::right_bottom) {
					newLTWH.z += dpos.x;
				}
				if (ht == boxHitTestResult::top || ht == boxHitTestResult::left_top || ht == boxHitTestResult::right_top) {
					newLTWH.y += dpos.y;
					newLTWH.w -= dpos.y;
				}
				if (ht == boxHitTestResult::bottom || ht == boxHitTestResult::left_bottom || ht == boxHitTestResult::right_bottom) {
					newLTWH.w += dpos.y;
				}
				if (input().key_down(key_code::left_shift) || input().key_down(key_code::right_shift)) {
					// maintain equal width/height
					if (newLTWH.z != newLTWH.w) {
						// TODO: this is suboptimal! (drag top left corner...)
						if (newLTWH.x != oldLTWH.x || newLTWH.z != oldLTWH.z) newLTWH.w = newLTWH.z; else newLTWH.z = newLTWH.w;
					}
				}
			}

			if (newLTWH.z <= 0 || newLTWH.w <= 0) newLTWH = oldLTWH;
			*(boxes[box]) = newLTWH;
		}

		prev_pos = pos;
		prev_lmb = lmb;
		prev_ht  = ht;
		prev_box = box;

		//auto res = context().main_window()->current_image().width();

		//GLFWwindow *glfwWin = context().main_window()->handle()->mHandle;
		//double xpos, ypos;
		//glfwGetCursorPos(glfwWin, &xpos, &ypos);

	}

	// Update the push constant data that will be used in render():
	void update() override
	{
		using namespace avk;
		using namespace gvk;

		handle_input();

		auto inFlightIndex = context().main_window()->in_flight_index_for_frame();
		const auto* quakeCamera = current_composition()->element_by_type<quake_camera>();
		assert(nullptr != quakeCamera);

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
		mTaaPushConstants.mUseYCoCg = mUseYCoCg;
		mTaaPushConstants.mDebugMode = mDebugMode;
		mTaaPushConstants.mDebugCenter = mDebugCenter;
		mTaaPushConstants.mDebugScale = mDebugScale;
		mTaaPushConstants.mVarianceClipping = mVarianceClipping;
		mTaaPushConstants.mLumaWeighting = mLumaWeighting;
		mTaaPushConstants.mVarClipGamma = mVarClipGamma;
		mTaaPushConstants.mMinAlpha = mMinAlpha;
		mTaaPushConstants.mMaxAlpha = mMaxAlpha;
		mTaaPushConstants.mRejectionAlpha = mRejectionAlpha;
		mTaaPushConstants.mRejectOutside = mRejectOutside;
		mTaaPushConstants.mUseVelocityVectors = mUseVelocityVectors;
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
		rdoc::beginSection(cmdbfr->handle(), "TAA pass");

		// ---------------------- If Anti-Aliasing is enabled perform the following actions --------------------------
		static bool isVeryFirstFrame = true;
		if (mTaaEnabled && !isVeryFirstFrame) {	// history is invalid for the very first frame

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
#if TAA_OUTPUT_IS_SRGB
				descriptor_binding(0, 3, *mResultImagesSrgb[inFlightLastIndex]),
#else
				descriptor_binding(0, 3, *mResultImages[inFlightLastIndex]),
#endif
				descriptor_binding(0, 4, *mSrcDepth[inFlightLastIndex]),
				descriptor_binding(0, 5, mResultImages[inFlightIndex]->as_storage_image()),
				descriptor_binding(0, 6, mDebugImages[inFlightIndex]->as_storage_image()),
				descriptor_binding(0, 7, *mSrcVelocity[inFlightIndex]),
				descriptor_binding(1, 0, mMatricesBuffer[inFlightIndex])
				}));
			cmdbfr->push_constants(mTaaPipeline->layout(), mTaaPushConstants);
			cmdbfr->handle().dispatch((mResultImages[inFlightIndex]->get_image().width() + 15u) / 16u, (mResultImages[inFlightIndex]->get_image().height() + 15u) / 16u, 1);


#if TAA_OUTPUT_IS_SRGB
			// Finally, copy into sRGB image:
			cmdbfr->establish_global_memory_barrier(
				pipeline_stage::compute_shader,                        /* -> */ pipeline_stage::transfer,
				memory_access::shader_buffers_and_images_write_access, /* -> */ memory_access::transfer_read_access
			);

			copy_image_to_another(mResultImages[inFlightIndex]->get_image(), mResultImagesSrgb[inFlightIndex]->get_image(), sync::with_barriers_into_existing_command_buffer(cmdbfr));

			cmdbfr->establish_global_memory_barrier(
				pipeline_stage::transfer,             /* -> */ pipeline_stage::transfer,
				memory_access::transfer_write_access, /* -> */ memory_access::transfer_read_access
			);
#endif

#if TAA_USE_POSTPROCESS_STEP
			if (mPostProcessEnabled) {
				// post-processing

				cmdbfr->establish_global_memory_barrier(
					pipeline_stage::compute_shader,                        /* -> */ pipeline_stage::compute_shader,
					memory_access::shader_buffers_and_images_write_access, /* -> */ memory_access::shader_buffers_and_images_any_access
				);

				cmdbfr->bind_pipeline(mPostProcessPipeline);
				cmdbfr->bind_descriptors(mPostProcessPipeline->layout(), mDescriptorCache.get_or_create_descriptor_sets({
					descriptor_binding(0, 1, *RESULT_IMAGES_MAYBESRGB[inFlightIndex]),
					descriptor_binding(0, 2, mPostProcessImages[inFlightIndex]->as_storage_image())
					}));
				cmdbfr->push_constants(mPostProcessPipeline->layout(), mPostProcessPushConstants);
				cmdbfr->handle().dispatch((mPostProcessImages[inFlightIndex]->get_image().width() + 15u) / 16u, (mPostProcessImages[inFlightIndex]->get_image().height() + 15u) / 16u, 1);
			}
#endif

			// Blit into backbuffer directly from here (ATTENTION if you'd like to render something in other invokees!)
			
			auto *p_final_image = &RESULT_IMAGES_MAYBESRGB[inFlightIndex]->get_image();
#if TAA_USE_POSTPROCESS_STEP
			if (mPostProcessEnabled) p_final_image = &mPostProcessImages[inFlightIndex]->get_image();
#endif
			auto &image_to_show = mShowDebug ? mDebugImages[inFlightIndex]->get_image() : *p_final_image;

			// TODO: is this barrier needed?
			cmdbfr->establish_global_memory_barrier(
				pipeline_stage::compute_shader,                        /* -> */ pipeline_stage::transfer,
				memory_access::shader_buffers_and_images_write_access, /* -> */ memory_access::transfer_read_access
			);

			blit_image(image_to_show, mainWnd->backbuffer_at_index(inFlightIndex).image_view_at(0)->get_image(), sync::with_barriers_into_existing_command_buffer(cmdbfr));

			helpers::record_timing_interval_end(cmdbfr->handle(), fmt::format("TAA {}", inFlightIndex));
		}
		// -------------------------- If Anti-Aliasing is disabled, do nothing but blit/copy ------------------------------
		else {
			// Blit into backbuffer directly from here (ATTENTION if you'd like to render something in other invokees!)
			blit_image(mSrcColor[inFlightIndex]->get_image(), mResultImages[inFlightIndex]->get_image(), sync::with_barriers_into_existing_command_buffer(cmdbfr));
#if TAA_OUTPUT_IS_SRGB
			blit_image(mSrcColor[inFlightIndex]->get_image(), mResultImagesSrgb[inFlightIndex]->get_image(), sync::with_barriers_into_existing_command_buffer(cmdbfr));
#endif
			blit_image(mSrcColor[inFlightIndex]->get_image(), mainWnd->backbuffer_at_index(inFlightIndex).image_view_at(0)->get_image(), sync::with_barriers_into_existing_command_buffer(cmdbfr));
		}

		rdoc::endSection(cmdbfr->handle());
		cmdbfr->end_recording();

		isVeryFirstFrame = false;

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
	bool mPostProcessEnabled = static_cast<bool>(TAA_USE_POSTPROCESS_STEP);
	int mColorClampingOrClipping = 1;
	bool mDepthCulling = false;
	bool mTextureLookupUnjitter = false;
	int mSampleDistribution = 1;
	float mAlpha = 0.05f; // 0.1f;
	bool mResetHistory = false;

	// Source color images per frame in flight:
	std::array<avk::image_view_t*, CF> mSrcColor;
	std::array<avk::image_view_t*, CF> mSrcDepth;
	std::array<avk::image_view_t*, CF> mSrcVelocity;
	// Destination images per frame in flight:
	std::array<avk::image_view, CF> mResultImages;
#if TAA_OUTPUT_IS_SRGB
	// Copying the result images into actually the same result images (but only sRGB format)
	// is a rather stupid workaround. It is also quite resource-intensive... Life's hard!
	std::array<avk::image_view, CF> mResultImagesSrgb;
#endif
	std::array<avk::image_view, CF> mDebugImages;
	std::array<avk::image_view, CF> mPostProcessImages;
	// For each history frame's image content, also store the associated projection matrix:
	std::array<glm::mat4, CF> mHistoryProjMatrices;
	std::array<glm::mat4, CF> mHistoryViewMatrices;
	std::array<avk::buffer, CF> mMatricesBuffer;

	avk::sampler mSampler;

	// Prepared command buffers to synchronize subsequent commands
	std::array<avk::command_buffer, CF> mSyncAfterCommandBuffers;

	avk::compute_pipeline mTaaPipeline;
	push_constants_for_taa mTaaPushConstants;

	avk::compute_pipeline mPostProcessPipeline;
	push_constants_for_postprocess mPostProcessPushConstants;


	// jitter debugging
	int mFixedJitterIndex = -1;
	float mJitterExtraScale = 1.0f;
	int mJitterSlowMotion = 1;
	float mJitterRotateDegrees = 0.f;
	bool mBypassHistoryUpdate = false; // used by slow motion

	int mDebugMode = 0; // 0=result, 1=color bb (rgb), 2=color bb(size), 3=history rejection
	bool mDebugCenter = false; // center result to vec4(0.5) - for showing vectors
	bool mShowDebug = false;
	float mDebugScale = 1.f;
	bool mUseYCoCg = false;
	bool mVarianceClipping = false;
	float mVarClipGamma = 1.0f;
	bool mLumaWeighting = false;
	float mMinAlpha = 1.0f - 0.97f;
	float mMaxAlpha = 1.0f - 0.88f;
	float mRejectionAlpha = 1.0f;
	bool mTriggerCapture = false;
	bool mRejectOutside = false;
	int mUseVelocityVectors = 1;		// 0=off 1=for movers only 2=for everything
};
