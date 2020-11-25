#pragma once

#include <gvk.hpp>
#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include "imgui_helper.hpp"
#include "helper_functions.hpp"

#include "rdoc_helper.hpp"
#include "IniUtil.h"
#include "shader_cpu_common.h"

// FidelityFX-CAS
#include <stdint.h>
#define A_CPU 1
#include "ffx_a.h"
#include "ffx_cas.h"


// This class handles the anti-aliasing post-processing effect(s).
// It is templated on the number of concurrent frames, i.e. some resources
// are created CF-times, once for each concurrent frame.
template <size_t CF>
class taa : public gvk::invokee
{
	// TODO: fix "non-shader parameters"
	struct Parameters {		// Note to self: be careful about alignment, esp. with vec#!
		float mAlpha						= 0.05f;
		int mColorClampingOrClipping		= 1;
		VkBool32 mDepthCulling				= VK_FALSE;
		VkBool32 mUnjitterNeighbourhood		= VK_FALSE;
		VkBool32 mUnjitterCurrentSample		= VK_FALSE;		// TODO: anything for depth/history depth??
		float mUnjitterFactor				= 1.0f;			// -1 or +1
		VkBool32 mPassThrough				= VK_FALSE;		// effectively disables TAA: result <- input,
		VkBool32 mUseYCoCg					= VK_FALSE;
		VkBool32 mShrinkChromaAxis			= VK_FALSE;		// ony used if mUseYCoCg - reduce Chroma influence on clip box
		VkBool32 mVarianceClipping			= VK_FALSE;
		VkBool32 mShapedNeighbourhood		= VK_FALSE;
		VkBool32 mLumaWeightingLottes		= VK_FALSE;;
		float mVarClipGamma					= 1.0f;
		float mMinAlpha						= 1.0f - 0.97f;	// used for luminance-based weighting
		float mMaxAlpha						= 1.0f - 0.88f;	// used for luminance-based weighting
		float mRejectionAlpha				= 1.0f;
		VkBool32 mRejectOutside				= VK_FALSE;
		int mUseVelocityVectors				= 1;			// 0=off 1=for movers only 2=for everything
		int mVelocitySampleMode				= 0;			// 0=simple 1=3x3_max 2=3x3_closest
		int mInterpolationMode				= 0;			// 0=bilinear 1=bicubic b-spline 2=bicubic catmull-rom
		VkBool32 mToneMapLumaKaris			= VK_FALSE;		// "tone mapping" by luma weighting (Karis)
		VkBool32 mAddNoise					= VK_FALSE;
		float mNoiseFactor					= 1.f / 510.f;	// small, way less than 1, e.g. 1/512
		VkBool32 mReduceBlendNearClamp		= VK_FALSE;		// reduce blend factor when near clamping (Karis14)
		int mDynamicAntiGhosting			= 0;			// dynamic anti-ghosting, inspired by Unreal Engine: 0=off 1=for movers only 2=for everything

		float pad1, pad2, pad3;
		// -- aligned again here
		glm::vec4 mDebugMask				= glm::vec4(1);
		int mDebugMode						= 0;			// 0=result, 1=color bb (rgb), 2=color bb(size), 3=history rejection;
		float mDebugScale					= 1.0f;
		VkBool32 mDebugCenter				= VK_FALSE;
		float final_pad1;
	};
	static_assert(sizeof(Parameters) % 16 == 0, "Parameters struct is not padded"); // very crude check for padding to 16-bytes

	//struct push_constants_for_taa {		// Note to self: be careful about alignment, esp. with vec#!
	//	// moved everything to uniforms buffer
	//};

	struct push_constants_for_sharpener {
		float sharpeningFactor	= 1.f;
	};

	struct push_constants_for_cas {
		varAU4(const0);
		varAU4(const1);
	};

	struct push_constants_for_postprocess {	// !ATTN to alignment!
		glm::ivec4 zoomSrcLTWH	= { 960 - 10, 540 - 10, 20, 20 };
		glm::ivec4 zoomDstLTWH	= { 1920 - 200 - 10, 10, 200, 200 };
		VkBool32 zoom			= false;
		VkBool32 showZoomBox	= true;
		int splitX = -1;
	};

	struct uniforms_for_taa {
		glm::mat4 mHistoryViewProjMatrix;
		glm::mat4 mInverseViewProjMatrix;

		Parameters	param[2];
		glm::vec4	mJitterNdc;		// only .xy used
		glm::vec4	mSinTime;		// sin(t/8), sin(t/4), sin(t/2), sin(t)
		VkBool32	splitScreen;
		int			splitX;
		VkBool32    mUpsampling;
		VkBool32    mBypassHistoryUpdate		= VK_FALSE;		// used by jitter debug slow motion
		VkBool32    mResetHistory				= VK_FALSE;

		float pad1,pad2,pad3;
	};
	static_assert(sizeof(uniforms_for_taa) % 16 == 0, "uniforms_for_taa struct is not padded"); // very crude check for padding to 16-bytes

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
	glm::vec2 get_jitter_offset_for_frame(gvk::window::frame_id_t aFrameId, std::vector<glm::vec2> *copyPatternDst = nullptr) const
	{
		//const static auto sResolution = gvk::context().main_window()->resolution();
		assert(mInputResolution.x);
		const static auto sResolution = mInputResolution;
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

		const float eighth = 1.f / 8.f;
		const static auto sRegular16SampleOffsets = avk::make_array<glm::vec2>(	// just for testing
			sPxSizeNDC * glm::vec2(-3.f*eighth, -3.f*eighth), sPxSizeNDC * glm::vec2(-1.f*eighth, -3.f*eighth), sPxSizeNDC * glm::vec2( 1.f*eighth, -3.f*eighth), sPxSizeNDC * glm::vec2( 3.f*eighth, -3.f*eighth),
			sPxSizeNDC * glm::vec2(-3.f*eighth, -1.f*eighth), sPxSizeNDC * glm::vec2(-1.f*eighth, -1.f*eighth), sPxSizeNDC * glm::vec2( 1.f*eighth, -1.f*eighth), sPxSizeNDC * glm::vec2( 3.f*eighth, -1.f*eighth),
			sPxSizeNDC * glm::vec2(-3.f*eighth,  1.f*eighth), sPxSizeNDC * glm::vec2(-1.f*eighth,  1.f*eighth), sPxSizeNDC * glm::vec2( 1.f*eighth,  1.f*eighth), sPxSizeNDC * glm::vec2( 3.f*eighth,  1.f*eighth),
			sPxSizeNDC * glm::vec2(-3.f*eighth,  3.f*eighth), sPxSizeNDC * glm::vec2(-1.f*eighth,  3.f*eighth), sPxSizeNDC * glm::vec2( 1.f*eighth,  3.f*eighth), sPxSizeNDC * glm::vec2( 3.f*eighth,  3.f*eighth)
			);


		// Select a specific distribution:
		const glm::vec2* sampleOffsetValues = nullptr;
		size_t numSampleOffsets = 0;
		glm::vec2 scaleBy = glm::vec2(1);
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
			sampleOffsetValues = sRegular16SampleOffsets.data();
			numSampleOffsets = sRegular16SampleOffsets.size();
			break;
		case 5:
			sampleOffsetValues = mDebugSampleOffsets.data();
			numSampleOffsets = mDebugSampleOffsets.size();
			scaleBy = sPxSizeNDC;
			break;
		}

		if (copyPatternDst) {
			copyPatternDst->resize(numSampleOffsets);
			for (auto i = 0; i < numSampleOffsets; ++i) copyPatternDst->at(i) = sampleOffsetValues[i] * scaleBy / sPxSizeNDC;
		}

		if (mJitterSlowMotion > 1) aFrameId /= mJitterSlowMotion;
		if (mFixedJitterIndex >= 0) aFrameId = mFixedJitterIndex;


		auto pos = sampleOffsetValues[aFrameId % numSampleOffsets] * scaleBy;

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
	void set_source_image_views(glm::uvec2 targetResolution, SRCCOLOR& aSourceColorImageViews, SRCD& aSourceDepthImageViews, SRCVELOCITY& aSourceVelocityImageViews)
	{
		std::vector<avk::command_buffer> layoutTransitions;

		for (size_t i = 0; i < CF; ++i) {
			// Store pointers to the source color result images
			if constexpr (std::is_pointer<typename SRCCOLOR::value_type>::value) {
				mSrcColor[i] = aSourceColorImageViews[i];
			} else {
				mSrcColor[i] = aSourceColorImageViews[i];
			}

			// Store pointers to the source depth images
			if constexpr (std::is_pointer<typename SRCD::value_type>::value) {
				mSrcDepth[i] = aSourceDepthImageViews[i];
			} else {
				mSrcDepth[i] = aSourceDepthImageViews[i];
			}

			// Store pointers to the source velocity result images
			if constexpr (std::is_pointer<typename SRCVELOCITY::value_type>::value) {
				mSrcVelocity[i] = aSourceVelocityImageViews[i];
			} else {
				mSrcVelocity[i] = aSourceVelocityImageViews[i];
			}


			//auto w = mSrcColor[i]->get_image().width();
			//auto h = mSrcColor[i]->get_image().height();
			auto w = targetResolution.x;
			auto h = targetResolution.y;

			mUpsampling = (w != mSrcColor[i]->get_image().width() || h != mSrcColor[i]->get_image().height());

			mResultImages[i] = gvk::context().create_image_view(
				gvk::context().create_image(w, h, TAA_IMAGE_FORMAT_RGB, 1, avk::memory_usage::device, avk::image_usage::general_storage_image)
			);
			rdoc::labelImage(mResultImages[i]->get_image().handle(), "taa.mResultImages", i);
			layoutTransitions.emplace_back(std::move(mResultImages[i]->get_image().transition_to_layout({}, avk::sync::with_barriers_by_return({}, {})).value()));

			mHistoryImages[i] = gvk::context().create_image_view(
				gvk::context().create_image(w, h, TAA_IMAGE_FORMAT_RGB, 1, avk::memory_usage::device, avk::image_usage::general_storage_image)
			);
			rdoc::labelImage(mHistoryImages[i]->get_image().handle(), "taa.mHistoryImages", i);
			layoutTransitions.emplace_back(std::move(mHistoryImages[i]->get_image().transition_to_layout({}, avk::sync::with_barriers_by_return({}, {})).value()));

			mTempImages[i] = gvk::context().create_image_view(
				gvk::context().create_image(w, h, TAA_IMAGE_FORMAT_RGB, 1, avk::memory_usage::device, avk::image_usage::general_storage_image)
			);
			rdoc::labelImage(mTempImages[i]->get_image().handle(), "taa.mTempImages", i);
			layoutTransitions.emplace_back(std::move(mTempImages[i]->get_image().transition_to_layout({}, avk::sync::with_barriers_by_return({}, {})).value()));

			mDebugImages[i] = gvk::context().create_image_view(
				gvk::context().create_image(w, h, vk::Format::eR16G16B16A16Sfloat, 1, avk::memory_usage::device, avk::image_usage::general_storage_image)
			);
			rdoc::labelImage(mDebugImages[i]->get_image().handle(), "taa.mDebugImages", i);
			layoutTransitions.emplace_back(std::move(mDebugImages[i]->get_image().transition_to_layout({}, avk::sync::with_barriers_by_return({}, {})).value()));

			mPostProcessImages[i] = gvk::context().create_image_view(
				gvk::context().create_image(w, h, TAA_IMAGE_FORMAT_POSTPROCESS, 1, avk::memory_usage::device, avk::image_usage::general_storage_image)
			);
			rdoc::labelImage(mPostProcessImages[i]->get_image().handle(), "taa.mPostProcessImages", i);
			layoutTransitions.emplace_back(std::move(mPostProcessImages[i]->get_image().transition_to_layout({}, avk::sync::with_barriers_by_return({}, {})).value()));

			mInputResolution = glm::uvec2(mSrcColor[0]->get_image().width(), mSrcColor[0]->get_image().height());
			mOutputResolution = targetResolution;
		}

		std::vector<avk::resource_reference<avk::command_buffer_t>> commandBufferReferences;
		std::transform(std::begin(layoutTransitions), std::end(layoutTransitions), std::back_inserter(commandBufferReferences), [](avk::command_buffer& cb) { return avk::referenced(*cb); });
		auto fen = mQueue->submit_with_fence(commandBufferReferences);
		fen->wait_until_signalled();

		// also initialize some gui elements based on image dimensions
		auto w = targetResolution.x;
		auto h = targetResolution.y;
		mSplitX = w / 2;
		int dZoomSrc = int(round(w / 96.f));
		int dZoomDst = int(round(w / 9.6f));
		int dZoomBrd = int(round(w / (96.f * 2.f)));
		mPostProcessPushConstants.zoomSrcLTWH = { (w - dZoomSrc) / 2, (h - dZoomSrc) / 2, dZoomSrc, dZoomSrc };
		mPostProcessPushConstants.zoomDstLTWH = { w - dZoomDst - dZoomBrd, dZoomBrd, dZoomDst, dZoomDst };
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

				if (!imgui_helper::globalEnable) return;

				const float checkbox_height = 23.f;
				const float combo_height    = 23.f;
				const float button_height   = 23.f;

				static bool copyParamsTo0 = false;
				static bool copyParamsTo1 = false;
				static bool switchParams  = false;
				for (int iPass = 0; iPass < 2; ++iPass) {
					bool isPrimary = (0 == iPass);
					if (!isPrimary && !mSplitScreen) break;

					Parameters &param = mParameters[iPass];

					Begin(iPass == 0 ? "Anti-Aliasing Settings" : "Anti-Aliasing Settings #2");
					SetWindowPos (ImVec2(270.0f + iPass * 400.f, 555.0f), ImGuiCond_FirstUseEver);
					SetWindowSize(ImVec2(220.0f, 130.0f),                 ImGuiCond_FirstUseEver);
					if (isPrimary) {
						Checkbox("enabled", &mTaaEnabled);
						SameLine(); if (Button("En&cap")) { mTaaEnabled = true; mTriggerCapture = true; }
						if (static_cast<float>(glfwGetTime()) - mLastResetHistoryTime < 0.5f) {
							SameLine(); TextColored(ImVec4(1, 0, 0, 1), "HISTORY RESET");
						}
					} else {
						SetCursorPosY(GetCursorPosY() + checkbox_height);
					}

					if (CollapsingHeader("Params")) {
						CheckboxB32("pass through", &param.mPassThrough); HelpMarker("Effectively disables TAA, but runs shader");
						static const char* sColorClampingClippingValues[] = { "nope", "clamp", "clip fast", "clip slow" };
						ComboW(120, "color clamp/clip", &param.mColorClampingOrClipping, sColorClampingClippingValues, IM_ARRAYSIZE(sColorClampingClippingValues));
						if (CheckboxB32("shaped neighbourhood", &param.mShapedNeighbourhood)) if (param.mShapedNeighbourhood) param.mVarianceClipping = VK_FALSE;
						HelpMarker("[Karis14] average the min/max of 3x3 and 5-tap clipboxes");
						if (CheckboxB32("variance clipping", &param.mVarianceClipping)) if (param.mVarianceClipping) param.mShapedNeighbourhood = VK_FALSE;
						SameLine(); SliderFloatW(100, "##gamma", &param.mVarClipGamma, 0.f, 2.f, "%.2f");
						CheckboxB32("use YCoCg", &param.mUseYCoCg);
						if (!param.mUseYCoCg) PushStyleVar(ImGuiStyleVar_Alpha, GetStyle().Alpha * 0.5f);
						SameLine(); CheckboxB32("shrink chroma", &param.mShrinkChromaAxis);
						if (!param.mUseYCoCg) PopStyleVar();
						HelpMarker("Use YCoCg: Perform clamping/clipping in YCoCg color space instead of RGB.\nShrink chroma: (in addition) Shape the YCoCg clip box to reduce the influence of chroma.");
						CheckboxB32("luma weight (Lottes)", &param.mLumaWeightingLottes); HelpMarker("Not to be confused with tonemap luma weight (Karis)!\nSet min and max alpha to define feedback range.");
						CheckboxB32("depth culling", &param.mDepthCulling);
						CheckboxB32("reject out-of-screen", &param.mRejectOutside);
						//Checkbox("texture lookup unjitter", &param.mTextureLookupUnjitter);
						CheckboxB32("unjitter neighbourhood",  &param.mUnjitterNeighbourhood);
						CheckboxB32("unjitter current sample", &param.mUnjitterCurrentSample);
						PushItemWidth(120);
						InputFloat("unjitter factor", &param.mUnjitterFactor);
					
						SliderFloat("alpha", &param.mAlpha, 0.0f, 1.0f);
						SliderFloat("a_min", &param.mMinAlpha, 0.0f, 1.0f); HelpMarker("Luma weighting min alpha");
						SliderFloat("a_max", &param.mMaxAlpha, 0.0f, 1.0f); HelpMarker("Luma weighting max alpha");
						SliderFloat("rejection alpha", &param.mRejectionAlpha, 0.0f, 1.0f);
						Combo("use velocity", &param.mUseVelocityVectors, "none\0movers\0all\0");
						Combo("vel.sampling", &param.mVelocitySampleMode, "simple\0""3x3 longest\0""3x3 closest\0");
						HelpMarker("simple:      just sample velocity at the current fragment\n"
								   "3x3 longest: take the longest velocity vector in a 3x3\n"
								   "             neighbourhood\n"
								   "3x3 closest: take the velocity from the (depth-wise) closest"
								   "             fragment in a 3x3 neighbourhood"
						);
						Combo("interpol", &param.mInterpolationMode, "bilinear\0bicubic b-Spline\0bicubic Catmull-Rom\0");
						PopItemWidth();

						CheckboxB32("tonemap luma w. (Karis)", &param.mToneMapLumaKaris);
						HelpMarker("Not to be confused with luma weight (Lottes)!");

						CheckboxB32("noise", &param.mAddNoise);
						SameLine();
						SliderFloat("##noisefac", &param.mNoiseFactor, 0.f, 0.01f, "%.4f", 2.f);

						CheckboxB32("reduce blend near clamp", &param.mReduceBlendNearClamp);
						HelpMarker("Anti-flicker: Reduce blend factor when history is near clamping [Karis14]");

						Combo("dynamic anti-ghosting", &param.mDynamicAntiGhosting, "off\0movers\0all\0");
						HelpMarker("Reject history if there is a no movement in a 5-tap neighbourhood and there was movement at the current pixel in the previous frame. [inspired by Unreal Engine]");

						if (isPrimary) {
							ComboW(120,"##sharpener", &mSharpener, "no sharpening\0simple sharpening\0FidelityFX-CAS\0");
							SameLine();
							SliderFloatW(80, "##sharpening factor", &mSharpenFactor, 0.f, 1.f, "%.1f");
						} else {
							SetCursorPosY(GetCursorPosY() + checkbox_height);
						}
					}

					if (isPrimary) { if (Button("reset history")) mResetHistory = true; } else SetCursorPosY(GetCursorPosY() + button_height);
					static const char* sDebugModeValues[] = { "color bb (rgb)", "color bb(size)", "rejection", "alpha", "velocity", "screen-result", "history-result", "debug" /* always last */ };
					if (isPrimary) Checkbox("debug##show debug", &mShowDebug); else Text("debug");
					SameLine();
					Combo("##debug mode", &param.mDebugMode, sDebugModeValues, IM_ARRAYSIZE(sDebugModeValues));
					PushItemWidth(100);
					SliderFloat("scale##debug scale", &param.mDebugScale, 0.f, 100.f, "%.0f");
					PopItemWidth();
					SameLine();
					CheckboxB32("center##debug center", &param.mDebugCenter);
					static bool mask_r, mask_g, mask_b;
					mask_r = param.mDebugMask.r != 0.f;
					mask_g = param.mDebugMask.g != 0.f;
					mask_b = param.mDebugMask.b != 0.f;
					Checkbox("R##debugmaskR", &mask_r); SameLine();
					Checkbox("G##debugmaskG", &mask_g); SameLine();
					Checkbox("B##debugmaskB", &mask_b);
					param.mDebugMask = glm::vec4(mask_r ? 1.f : 0.f, mask_g ? 1.f : 0.f, mask_b ? 1.f : 0.f, 1.f);

					if (isPrimary) {

						if (CollapsingHeader("Split screen")) {
							Checkbox("split", &mSplitScreen); SameLine();
							PushItemWidth(60);
							InputInt("##split x", &mSplitX, 0);
							PopItemWidth();
							Text("params:"); SameLine();
							copyParamsTo1 = Button("1->2"); SameLine();
							copyParamsTo0 = Button("1<-2"); SameLine();
							switchParams = Button("flip");
						}

						if (CollapsingHeader("Jitter")) {
							PushID("JitterStuff");
							static const char* sSampleDistributionValues[] = { "circular quad", "uniform4 helix", "halton x8", "halton x16", "grid x16", "debug" };
							if (isPrimary) Combo("sample pattern", &mSampleDistribution, sSampleDistributionValues, IM_ARRAYSIZE(sSampleDistributionValues)); else SetCursorPosY(GetCursorPosY() + combo_height);

							SliderInt("lock", &mFixedJitterIndex, -1, 16);
							InputFloat("scale", &mJitterExtraScale, 0.f, 0.f, "%.2f");
							InputInt("slowdown", &mJitterSlowMotion);
							InputFloat("rotate", &mJitterRotateDegrees);
							Text("Debug pattern:");
							SameLine();
							if (Button("copy")) {
								std::vector<glm::vec2> tmp;
								get_jitter_offset_for_frame(0, &tmp);
								mDebugSampleOffsets = tmp;
							}
							int delAt = -1;
							int addAt = -1;
							for (int i = 0; i < static_cast<int>(mDebugSampleOffsets.size()); ++i) {
								PushID(i);
								Text("#%d", i); SameLine();
								InputFloat2W(100, "##pos", &mDebugSampleOffsets[i].x, "%.2f");
								SameLine(); if (Button("+")) addAt = i;
								SameLine(); if (Button("-")) delAt = i;
								PopID();
							}
							if (delAt >= 0 && mDebugSampleOffsets.size() < 2) delAt = -1; // keep at least one sample
							if (addAt >= 0) mDebugSampleOffsets.insert(mDebugSampleOffsets.begin() + addAt + 1, glm::vec2(0));
							if (delAt >= 0) mDebugSampleOffsets.erase(mDebugSampleOffsets.begin() + delAt);
							PopID();
						}

						if (CollapsingHeader("Postprocess")) {
							auto &pp = mPostProcessPushConstants;

							static glm::ivec4 orig_zoomsrc = pp.zoomSrcLTWH;
							static glm::ivec4 orig_zoomdst = pp.zoomDstLTWH;

							Checkbox("enable", &mPostProcessEnabled);
							CheckboxB32("zoom", &pp.zoom);	
							SameLine(); CheckboxB32("show box", &pp.showZoomBox);
							SameLine(); if (Button("rst")) { pp.zoomSrcLTWH = orig_zoomsrc; pp.zoomDstLTWH = orig_zoomdst; }
							InputInt4("src", &pp.zoomSrcLTWH.x); HelpMarker("Left/Top/Width/Height");
							InputInt4("dst", &pp.zoomDstLTWH.x);

						}

						Checkbox("Reset history at any change", &mResetHistoryOnChange);
					}

					End();
				}

				if (copyParamsTo0) mParameters[0] = mParameters[1];
				if (copyParamsTo1) mParameters[1] = mParameters[0];
				if (switchParams ) std::swap(mParameters[0], mParameters[1]);
				copyParamsTo0 = copyParamsTo1 = switchParams = false;

				});
		} else {
			LOG_WARNING("No component of type cgb::imgui_manager found => could not install ImGui callback.");
		}
	}

	// called from main
	void init_updater(gvk::updater &updater) {
		LOG_DEBUG("TAA: initing updater");
		std::vector<avk::compute_pipeline *> comp_pipes = { &mTaaPipeline, &mSharpenerPipeline, &mCasPipeline, &mPostProcessPipeline };
		for (auto ppipe : comp_pipes) {
			ppipe->enable_shared_ownership();
			updater.on(gvk::shader_files_changed_event(*ppipe)).update(*ppipe);
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

		mSampler = context().create_sampler(avk::filter_mode::bilinear, avk::border_handling_mode::clamp_to_edge, 0);	// ac: changed from clamp_to_border to clamp_to_edge

		for (size_t i = 0; i < CF; ++i) {
			mUniformsBuffer[i] = context().create_buffer(memory_usage::host_coherent, {}, uniform_buffer_meta::create_from_size(sizeof(uniforms_for_taa)));
		}

		mTaaPipeline = context().create_compute_pipeline_for(
			compute_shader("shaders/taa.comp.spv"),
			descriptor_binding(0, 0, mSampler),
			descriptor_binding(0, 1, *mSrcColor[0]),
			descriptor_binding(0, 2, *mSrcDepth[0]),
			descriptor_binding(0, 3, *mResultImages[0]),
			descriptor_binding(0, 4, *mSrcDepth[0]),
			descriptor_binding(0, 5, mResultImages[0]->as_storage_image()),		// output for screen
			descriptor_binding(0, 6, mDebugImages[0]->as_storage_image()),
			descriptor_binding(0, 7, *mSrcVelocity[0]),
			descriptor_binding(0, 8, mHistoryImages[0]->as_storage_image()),	// output for history
			descriptor_binding(1, 0, mUniformsBuffer[0])
			//push_constant_binding_data{ shader_type::compute, 0, sizeof(push_constants_for_taa) }
		);

		mSharpenerPipeline = context().create_compute_pipeline_for(
			compute_shader("shaders/sharpen.comp.spv"),
			descriptor_binding(0, 1, *mResultImages[0]),
			descriptor_binding(0, 2, mTempImages[0]->as_storage_image()),
			push_constant_binding_data{ shader_type::compute, 0, sizeof(push_constants_for_sharpener) }
		);

		mCasPipeline = context().create_compute_pipeline_for(
			compute_shader("shaders/sharpen_cas.comp.spv"),
			descriptor_binding(0, 1, mResultImages[0]->as_storage_image()),
			descriptor_binding(0, 2, mTempImages[0]->as_storage_image()),
			push_constant_binding_data{ shader_type::compute, 0, sizeof(push_constants_for_cas) }
		);

		mPostProcessPipeline = context().create_compute_pipeline_for(
			compute_shader("shaders/post_process.comp.spv"),
			descriptor_binding(0, 1, *mResultImages[0]),
			descriptor_binding(0, 2, mPostProcessImages[0]->as_storage_image()),
			push_constant_binding_data{ shader_type::compute, 0, sizeof(push_constants_for_postprocess) }
		);

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

		ImGuiIO &io = ImGui::GetIO();

		if (!io.WantTextInput && input().key_pressed(key_code::r)) mResetHistory = true;

		const auto* quakeCamera = current_composition()->element_by_type<quake_camera>();
		if (quakeCamera && quakeCamera->is_enabled()) return;

		if (io.WantCaptureMouse) return;

		if (!mPostProcessPushConstants.zoom && !mSplitScreen) return;

		static glm::ivec2 prev_pos = {0,0};
		static bool prev_lmb = false;
		static boxHitTestResult prev_ht = boxHitTestResult::outside;
		static int prev_box = -1; // -1 = none, -2 = splitter

		static std::vector<glm::ivec4 *> boxes = { &mPostProcessPushConstants.zoomSrcLTWH, &mPostProcessPushConstants.zoomDstLTWH };

		glm::ivec2 pos = glm::ivec2(input().cursor_position());
		bool lmb = (input().mouse_button_down(GLFW_MOUSE_BUTTON_LEFT));
		bool click = lmb && !prev_lmb;
		bool drag  = lmb && prev_lmb;
		
		// when dragging, keep previous hit-test
		auto ht = prev_ht;
		int box = prev_box;
		if (!drag) {

			ht = boxHitTestResult::outside; box = -1;
			if (mPostProcessPushConstants.zoom) {
				for (auto i = 0; i < boxes.size(); ++i) {
					auto ht_i = boxHitTest(pos, *(boxes[i]), true);
					if (ht_i != boxHitTestResult::outside) {
						ht = ht_i; box = i; break;
					}
				}
			}
			if (mSplitScreen && ht == boxHitTestResult::outside) {
				if (pos.x >= mSplitX-2 && pos.x <= mSplitX+2) {
					ht = boxHitTestResult::left; box = -2;
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
		} else if (lmb && box == -2) {
			mSplitX = pos.x;
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
		bool bypassHistUpdate = false;
		if (mJitterSlowMotion > 1) {
			static gvk::window::frame_id_t lastJitterIndex = 0;
			auto thisJitterIndex = gvk::context().main_window()->current_frame() / mJitterSlowMotion;
			bypassHistUpdate = (thisJitterIndex == lastJitterIndex);
			lastJitterIndex = thisJitterIndex;
		}

		// reset history on (any) parameter change?
		static Parameters oldParams[2] = {};
		static bool oldParamsValid = false;
		if (mResetHistoryOnChange && oldParamsValid) {
			for (int i = 0; i < 2; ++i) {
				// zero out stuff where we do NOT want a history reset
				Parameters compare[2] = { mParameters[i], oldParams[i] };
				for (int j = 0; j < 2; ++j) {
					compare[j].mDebugMode   = 0;
					compare[j].mDebugScale  = 0.f;
					compare[j].mDebugCenter = VK_FALSE;
				}

				if (memcmp(&compare[0], &compare[1], sizeof(Parameters)) != 0) mResetHistory = true;
			}
		}

		if (mResetHistory) mLastResetHistoryTime = static_cast<float>(glfwGetTime());

		mHistoryViewMatrices[inFlightIndex] = quakeCamera->view_matrix();
		const auto jitter = get_jitter_offset_for_frame(gvk::context().main_window()->current_frame());
		//printf("jitter %f %f\n", jitter.x, jitter.y);
		for (int i = 0; i < 2; ++i) {
			mTaaUniforms.param[i] = mParameters[i];
		}
		mTaaUniforms.mJitterNdc  = glm::vec4(jitter.x, jitter.y, 0.f, 0.f);
		mTaaUniforms.mSinTime    = glm::sin(glm::vec4(.125f, .25f, .5f, 1.f) * static_cast<float>(glfwGetTime()));
		mTaaUniforms.mUpsampling = mUpsampling;
		mTaaUniforms.splitScreen = mSplitScreen;
		mTaaUniforms.splitX      = mSplitX;
		mTaaUniforms.mBypassHistoryUpdate = bypassHistUpdate;
		mTaaUniforms.mResetHistory = mResetHistory;

		mPostProcessPushConstants.splitX = mSplitScreen ? mSplitX : -1;

		static float prevSharpenFactor = -1.f;
		if (mSharpenFactor != prevSharpenFactor) {
			prevSharpenFactor = mSharpenFactor;
			mSharpenerPushConstants.sharpeningFactor = mSharpenFactor;
			CasSetup(mCasPushConstants.const0, mCasPushConstants.const1, mSharpenFactor, AF1(mOutputResolution.x), AF1(mOutputResolution.y), AF1(mOutputResolution.x), AF1(mOutputResolution.y));
		}

		mResetHistory = false;
		for (int i = 0; i < 2; ++i) oldParams[i] = mParameters[i];
		oldParamsValid = true;
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

			// fill in matrices to uniforms UBO
			mTaaUniforms.mInverseViewProjMatrix = glm::inverse(mHistoryProjMatrices[inFlightIndex] * mHistoryViewMatrices[inFlightIndex]);
			mTaaUniforms.mHistoryViewProjMatrix = mHistoryProjMatrices[inFlightLastIndex] * mHistoryViewMatrices[inFlightLastIndex];
			mUniformsBuffer[inFlightIndex]->fill(&mTaaUniforms, 0, sync::not_required()); // sync is done with establish_global_memory_barrier below

			helpers::record_timing_interval_start(cmdbfr->handle(), fmt::format("TAA {}", inFlightIndex));

			cmdbfr->establish_global_memory_barrier(
				pipeline_stage::transfer,             /* -> */ pipeline_stage::compute_shader,
				memory_access::transfer_write_access, /* -> */ memory_access::shader_buffers_and_images_read_access
			);

			// We are going to use the previous in flight frame as history for TAA
			static_assert(CF > 1);

			// Apply Temporal Anti-Aliasing:
			cmdbfr->bind_pipeline(const_referenced(mTaaPipeline));
			cmdbfr->bind_descriptors(mTaaPipeline->layout(), mDescriptorCache.get_or_create_descriptor_sets({
				descriptor_binding(0, 0, mSampler),
				descriptor_binding(0, 1, *mSrcColor[inFlightIndex]),							// -> shader: uCurrentFrame
				descriptor_binding(0, 2, *mSrcDepth[inFlightIndex]),							// -> shader: uCurrentDepth
				descriptor_binding(0, 3, *mHistoryImages[inFlightLastIndex]),					// -> shader: uHistoryFrame
				descriptor_binding(0, 4, *mSrcDepth[inFlightLastIndex]),						// -> shader: uHistoryDepth
				descriptor_binding(0, 5, mResultImages[inFlightIndex]->as_storage_image()),		// -> shader: uResultScreen
				descriptor_binding(0, 6, mDebugImages[inFlightIndex]->as_storage_image()),		// -> shader: uDebug
				descriptor_binding(0, 7, *mSrcVelocity[inFlightIndex]),							// -> shader: uCurrentVelocity
				descriptor_binding(0, 8, mHistoryImages[inFlightIndex]->as_storage_image()),	// -> shader: uResultHistory
				descriptor_binding(1, 0, mUniformsBuffer[inFlightIndex])
				}));
			//cmdbfr->push_constants(mTaaPipeline->layout(), mTaaPushConstants);
			cmdbfr->handle().dispatch((mResultImages[inFlightIndex]->get_image().width() + 15u) / 16u, (mResultImages[inFlightIndex]->get_image().height() + 15u) / 16u, 1);


			auto pLastProducedImageView = &mResultImages[inFlightIndex];

			if (mSharpener) {
				cmdbfr->establish_global_memory_barrier(
					pipeline_stage::compute_shader,                        /* -> */ pipeline_stage::compute_shader,
					memory_access::shader_buffers_and_images_write_access, /* -> */ memory_access::shader_buffers_and_images_any_access
				);

				if (mSharpener == 1) {
					cmdbfr->bind_pipeline(const_referenced(mSharpenerPipeline));
					cmdbfr->bind_descriptors(mSharpenerPipeline->layout(), mDescriptorCache.get_or_create_descriptor_sets({
						descriptor_binding(0, 1, *mResultImages[inFlightIndex]),
						descriptor_binding(0, 2, mTempImages[inFlightIndex]->as_storage_image())
						}));

					cmdbfr->push_constants(mSharpenerPipeline->layout(), mSharpenerPushConstants);
					cmdbfr->handle().dispatch((mTempImages[inFlightIndex]->get_image().width() + 15u) / 16u, (mTempImages[inFlightIndex]->get_image().height() + 15u) / 16u, 1);
				} else {
					cmdbfr->bind_pipeline(const_referenced(mCasPipeline));
					cmdbfr->bind_descriptors(mCasPipeline->layout(), mDescriptorCache.get_or_create_descriptor_sets({
						descriptor_binding(0, 1, mResultImages[inFlightIndex]->as_storage_image()),
						descriptor_binding(0, 2, mTempImages[inFlightIndex]->as_storage_image())
						}));

					cmdbfr->push_constants(mCasPipeline->layout(), mCasPushConstants);
					cmdbfr->handle().dispatch((mTempImages[inFlightIndex]->get_image().width() + 15u) / 16u, (mTempImages[inFlightIndex]->get_image().height() + 15u) / 16u, 1);
				}

				pLastProducedImageView = &mTempImages[inFlightIndex];
			}

			if (mShowDebug) pLastProducedImageView = &mDebugImages[inFlightIndex]; // apply post-processing to debug image, so we can use zoom there too

			if (mPostProcessEnabled) {
				// post-processing

				cmdbfr->establish_global_memory_barrier(
					pipeline_stage::compute_shader,                        /* -> */ pipeline_stage::compute_shader,
					memory_access::shader_buffers_and_images_write_access, /* -> */ memory_access::shader_buffers_and_images_any_access
				);

				cmdbfr->bind_pipeline(const_referenced(mPostProcessPipeline));
				cmdbfr->bind_descriptors(mPostProcessPipeline->layout(), mDescriptorCache.get_or_create_descriptor_sets({
					descriptor_binding(0, 1, **pLastProducedImageView),
					descriptor_binding(0, 2, mPostProcessImages[inFlightIndex]->as_storage_image())
					}));
				cmdbfr->push_constants(mPostProcessPipeline->layout(), mPostProcessPushConstants);
				cmdbfr->handle().dispatch((mPostProcessImages[inFlightIndex]->get_image().width() + 15u) / 16u, (mPostProcessImages[inFlightIndex]->get_image().height() + 15u) / 16u, 1);

				pLastProducedImageView = &mPostProcessImages[inFlightIndex];
			}

			auto &image_to_show = (*pLastProducedImageView)->get_image();

			// TODO: is this barrier needed?
			cmdbfr->establish_global_memory_barrier(
				pipeline_stage::compute_shader,                        /* -> */ pipeline_stage::transfer,
				memory_access::shader_buffers_and_images_write_access, /* -> */ memory_access::transfer_read_access
			);

			blit_image(image_to_show, mainWnd->backbuffer_at_index(inFlightIndex)->image_at(0), sync::with_barriers_into_existing_command_buffer(*cmdbfr));

			helpers::record_timing_interval_end(cmdbfr->handle(), fmt::format("TAA {}", inFlightIndex));
		}
		// -------------------------- If Anti-Aliasing is disabled, do nothing but blit/copy ------------------------------
		else {
			// Blit into backbuffer directly from here (ATTENTION if you'd like to render something in other invokees!)
			blit_image(mSrcColor[inFlightIndex]->get_image(), mResultImages[inFlightIndex]->get_image(), sync::with_barriers_into_existing_command_buffer(*cmdbfr));
			blit_image(mSrcColor[inFlightIndex]->get_image(), mainWnd->backbuffer_at_index(inFlightIndex)->image_at(0), sync::with_barriers_into_existing_command_buffer(*cmdbfr));
		}

		rdoc::endSection(cmdbfr->handle());
		cmdbfr->end_recording();

		isVeryFirstFrame = false;

		// The swap chain provides us with an "image available semaphore" for the current frame.
		// Only after the swapchain image has become available, we may start rendering into it.
		auto imageAvailableSemaphore = mainWnd->consume_current_image_available_semaphore();

		// Submit the draw call and take care of the command buffer's lifetime:
		mQueue->submit(cmdbfr, imageAvailableSemaphore);
		mainWnd->handle_lifetime(std::move(cmdbfr));
	}

	void finalize() override
	{
	}

	void writeSettingsToIni(mINI::INIStructure &ini) {
		std::string sec;
		
		for (int iPass = 0; iPass < 2; ++iPass) {
			sec = "TAA_Param_" + std::to_string(iPass);
			Parameters &param = mParameters[iPass];

			iniWriteBool32	(ini, sec, "mPassThrough",				param.mPassThrough);
			iniWriteInt		(ini, sec, "mColorClampingOrClipping",	param.mColorClampingOrClipping);
			iniWriteBool32	(ini, sec, "mShapedNeighbourhood",		param.mShapedNeighbourhood);
			iniWriteBool32	(ini, sec, "mVarianceClipping",			param.mVarianceClipping);
			iniWriteFloat	(ini, sec, "mVarClipGamma",				param.mVarClipGamma);
			iniWriteBool32	(ini, sec, "mUseYCoCg",					param.mUseYCoCg);
			iniWriteBool32	(ini, sec, "mLumaWeightingLottes",		param.mLumaWeightingLottes);
			iniWriteBool32	(ini, sec, "mDepthCulling",				param.mDepthCulling);
			iniWriteBool32	(ini, sec, "mRejectOutside",			param.mRejectOutside);
			iniWriteBool32	(ini, sec, "mUnjitterNeighbourhood",	param.mUnjitterNeighbourhood);
			iniWriteBool32	(ini, sec, "mUnjitterCurrentSample",	param.mUnjitterCurrentSample);
			iniWriteFloat	(ini, sec, "mUnjitterFactor",			param.mUnjitterFactor);
			iniWriteFloat	(ini, sec, "mAlpha",					param.mAlpha);
			iniWriteFloat	(ini, sec, "mMinAlpha", 				param.mMinAlpha);
			iniWriteFloat	(ini, sec, "mMaxAlpha", 				param.mMaxAlpha);
			iniWriteFloat	(ini, sec, "mRejectionAlpha",			param.mRejectionAlpha);
			iniWriteInt		(ini, sec, "mUseVelocityVectors",		param.mUseVelocityVectors);
			iniWriteInt		(ini, sec, "mVelocitySampleMode",		param.mVelocitySampleMode);
			iniWriteInt		(ini, sec, "mInterpolationMode",		param.mInterpolationMode);
			iniWriteBool32	(ini, sec, "mToneMapLumaKaris",			param.mToneMapLumaKaris);
			iniWriteBool32	(ini, sec, "mAddNoise",					param.mAddNoise);
			iniWriteFloat	(ini, sec, "mNoiseFactor",				param.mNoiseFactor);
			iniWriteBool32	(ini, sec, "mReduceBlendNearClamp",		param.mReduceBlendNearClamp);
			iniWriteInt		(ini, sec, "mDynamicAntiGhosting",		param.mDynamicAntiGhosting);
			iniWriteVec4	(ini, sec, "mDebugMask",				param.mDebugMask);
			iniWriteInt		(ini, sec, "mDebugMode",				param.mDebugMode);
			iniWriteFloat	(ini, sec, "mDebugScale",				param.mDebugScale);
			iniWriteBool32	(ini, sec, "mDebugCenter",				param.mDebugCenter);
		}

		sec = "TAA_Primary";
		iniWriteBool	(ini, sec, "mTaaEnabled",					mTaaEnabled);
		iniWriteInt		(ini, sec, "mSampleDistribution",			mSampleDistribution);
		iniWriteInt		(ini, sec, "mSharpener",					mSharpener);
		iniWriteFloat	(ini, sec, "mSharpenFactor",				mSharpenFactor);
		iniWriteBool	(ini, sec, "mShowDebug",					mShowDebug);
		iniWriteBool	(ini, sec, "mSplitScreen",					mSplitScreen);
		iniWriteInt		(ini, sec, "mSplitX",						mSplitX);
		iniWriteInt		(ini, sec, "mFixedJitterIndex",				mFixedJitterIndex);
		iniWriteFloat	(ini, sec, "mJitterExtraScale",				mJitterExtraScale);
		iniWriteInt		(ini, sec, "mJitterSlowMotion",				mJitterSlowMotion);
		iniWriteFloat	(ini, sec, "mJitterRotateDegrees",			mJitterRotateDegrees);
		iniWriteBool	(ini, sec, "mResetHistoryOnChange",			mResetHistoryOnChange);

		iniWriteInt		(ini, sec, "mDebugSampleOffsets.size",		static_cast<int>(mDebugSampleOffsets.size()));
		for (int i = 0; i < static_cast<int>(mDebugSampleOffsets.size()); ++i) {
			iniWriteVec2(ini, sec, "mDebugSampleOffsets_" + std::to_string(i),	mDebugSampleOffsets[i]);
		}

		sec = "TAA_Postprocess";
		auto &pp = mPostProcessPushConstants;
		iniWriteBool	(ini, sec, "mPostProcessEnabled",			mPostProcessEnabled);
		iniWriteBool32	(ini, sec, "zoom",							pp.zoom);
		iniWriteBool32	(ini, sec, "showZoomBox",					pp.showZoomBox);
		iniWriteIVec4	(ini, sec, "zoomSrcLTWH",					pp.zoomSrcLTWH);
		iniWriteIVec4	(ini, sec, "zoomDstLTWH",					pp.zoomDstLTWH);
	}

	void readSettingsFromIni(mINI::INIStructure &ini) {
		std::string sec;

		for (int iPass = 0; iPass < 2; ++iPass) {
			sec = "TAA_Param_" + std::to_string(iPass);
			Parameters &param = mParameters[iPass];

			iniReadBool32	(ini, sec, "mPassThrough",				param.mPassThrough);
			iniReadInt		(ini, sec, "mColorClampingOrClipping",	param.mColorClampingOrClipping);
			iniReadBool32	(ini, sec, "mShapedNeighbourhood",		param.mShapedNeighbourhood);
			iniReadBool32	(ini, sec, "mVarianceClipping",			param.mVarianceClipping);
			iniReadFloat	(ini, sec, "mVarClipGamma",				param.mVarClipGamma);
			iniReadBool32	(ini, sec, "mUseYCoCg",					param.mUseYCoCg);
			iniReadBool32	(ini, sec, "mLumaWeightingLottes",		param.mLumaWeightingLottes);
			iniReadBool32	(ini, sec, "mDepthCulling",				param.mDepthCulling);
			iniReadBool32	(ini, sec, "mRejectOutside",			param.mRejectOutside);
			iniReadBool32	(ini, sec, "mUnjitterNeighbourhood",	param.mUnjitterNeighbourhood);
			iniReadBool32	(ini, sec, "mUnjitterCurrentSample",	param.mUnjitterCurrentSample);
			iniReadFloat	(ini, sec, "mUnjitterFactor",			param.mUnjitterFactor);
			iniReadFloat	(ini, sec, "mAlpha",					param.mAlpha);
			iniReadFloat	(ini, sec, "mMinAlpha", 				param.mMinAlpha);
			iniReadFloat	(ini, sec, "mMaxAlpha", 				param.mMaxAlpha);
			iniReadFloat	(ini, sec, "mRejectionAlpha",			param.mRejectionAlpha);
			iniReadInt		(ini, sec, "mUseVelocityVectors",		param.mUseVelocityVectors);
			iniReadInt		(ini, sec, "mVelocitySampleMode",		param.mVelocitySampleMode);
			iniReadInt		(ini, sec, "mInterpolationMode",		param.mInterpolationMode);
			iniReadBool32	(ini, sec, "mToneMapLumaKaris",			param.mToneMapLumaKaris);
			iniReadBool32	(ini, sec, "mAddNoise",					param.mAddNoise);
			iniReadFloat	(ini, sec, "mNoiseFactor",				param.mNoiseFactor);
			iniReadBool32	(ini, sec, "mReduceBlendNearClamp",		param.mReduceBlendNearClamp);
			iniReadInt		(ini, sec, "mDynamicAntiGhosting",		param.mDynamicAntiGhosting);
			iniReadVec4		(ini, sec, "mDebugMask",				param.mDebugMask);
			iniReadInt		(ini, sec, "mDebugMode",				param.mDebugMode);
			iniReadFloat	(ini, sec, "mDebugScale",				param.mDebugScale);
			iniReadBool32	(ini, sec, "mDebugCenter",				param.mDebugCenter);
		}

		sec = "TAA_Primary";
		iniReadBool		(ini, sec, "mTaaEnabled",					mTaaEnabled);
		iniReadInt		(ini, sec, "mSampleDistribution",			mSampleDistribution);
		iniReadInt		(ini, sec, "mSharpener",					mSharpener);
		iniReadFloat	(ini, sec, "mSharpenFactor",				mSharpenFactor);
		iniReadBool		(ini, sec, "mShowDebug",					mShowDebug);
		iniReadBool		(ini, sec, "mSplitScreen",					mSplitScreen);
		iniReadInt		(ini, sec, "mSplitX",						mSplitX);
		iniReadInt		(ini, sec, "mFixedJitterIndex",				mFixedJitterIndex);
		iniReadFloat	(ini, sec, "mJitterExtraScale",				mJitterExtraScale);
		iniReadInt		(ini, sec, "mJitterSlowMotion",				mJitterSlowMotion);
		iniReadFloat	(ini, sec, "mJitterRotateDegrees",			mJitterRotateDegrees);
		iniReadBool		(ini, sec, "mResetHistoryOnChange",			mResetHistoryOnChange);

		int nSamples = 0;
		iniReadInt		(ini, sec, "mDebugSampleOffsets.size",		nSamples);
		mDebugSampleOffsets.resize(std::max(1, nSamples), glm::vec2(0));
		for (int i = 0; i < nSamples; ++i) {
			iniReadVec2 (ini, sec, "mDebugSampleOffsets_" + std::to_string(i),	mDebugSampleOffsets[i]);
		}

		sec = "TAA_Postprocess";
		auto &pp = mPostProcessPushConstants;
		iniReadBool		(ini, sec, "mPostProcessEnabled",			mPostProcessEnabled);
		iniReadBool32	(ini, sec, "zoom",							pp.zoom);
		iniReadBool32	(ini, sec, "showZoomBox",					pp.showZoomBox);
		iniReadIVec4	(ini, sec, "zoomSrcLTWH",					pp.zoomSrcLTWH);
		iniReadIVec4	(ini, sec, "zoomDstLTWH",					pp.zoomDstLTWH);
	}

private:
	avk::queue* mQueue;
	avk::descriptor_cache mDescriptorCache;

	// Settings, which can be modified via ImGui:
	bool mTaaEnabled = true;
	bool mPostProcessEnabled = true;
	int mSampleDistribution = 1;
	bool mResetHistory = false;

	// Source color images per frame in flight:
	std::array<avk::image_view_t*, CF> mSrcColor;
	std::array<avk::image_view_t*, CF> mSrcDepth;
	std::array<avk::image_view_t*, CF> mSrcVelocity;
	// Destination images per frame in flight:
	std::array<avk::image_view, CF> mResultImages;
	std::array<avk::image_view, CF> mTempImages;
	std::array<avk::image_view, CF> mDebugImages;
	std::array<avk::image_view, CF> mPostProcessImages;
	std::array<avk::image_view, CF> mHistoryImages;			// use a separate history buffer for now, makes life easier..

	// For each history frame's image content, also store the associated projection matrix:
	std::array<glm::mat4, CF> mHistoryProjMatrices;
	std::array<glm::mat4, CF> mHistoryViewMatrices;
	std::array<avk::buffer, CF> mUniformsBuffer;

	avk::sampler mSampler;

	// Prepared command buffers to synchronize subsequent commands
	std::array<avk::command_buffer, CF> mSyncAfterCommandBuffers;

	avk::compute_pipeline mTaaPipeline;
	//push_constants_for_taa mTaaPushConstants;
	uniforms_for_taa mTaaUniforms;

	avk::compute_pipeline mPostProcessPipeline;
	push_constants_for_postprocess mPostProcessPushConstants;

	avk::compute_pipeline mSharpenerPipeline;
	push_constants_for_sharpener mSharpenerPushConstants;

	avk::compute_pipeline mCasPipeline;
	push_constants_for_cas mCasPushConstants;

	Parameters mParameters[2];

	// jitter debugging
	int mFixedJitterIndex = -1;
	float mJitterExtraScale = 1.0f;
	int mJitterSlowMotion = 1;
	float mJitterRotateDegrees = 0.f;

	bool mShowDebug = false;
	bool mTriggerCapture = false;

	bool mSplitScreen = false;
	int  mSplitX = 0;

	bool mUpsampling = false;
	glm::uvec2 mInputResolution  = {};	// lo res
	glm::uvec2 mOutputResolution = {};	// hi res

	int mSharpener = 0; // 0=off 1=simple 2=FidelityFX-CAS
	float mSharpenFactor = 0.5f;

	bool mResetHistoryOnChange = true; // reset history when any parameter has changed?
	float mLastResetHistoryTime = 0.f;

	std::vector<glm::vec2> mDebugSampleOffsets = { {0.f, 0.f} };
};
