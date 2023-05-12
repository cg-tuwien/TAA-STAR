#pragma once

#include <imgui.h>

namespace imgui_helper {
	bool globalEnable = true;

	static void HelpMarker(const char* desc, bool sameLine = true) {
		if (sameLine) ImGui::SameLine();

		ImGui::TextDisabled("(?)");
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
			ImGui::TextUnformatted(desc);
			ImGui::PopTextWrapPos();
			ImGui::EndTooltip();
		}
	}

	// checkbox for VkBool32
	static bool CheckboxB32(const char* label, uint32_t* v) {
		bool b = (0u != *v);
		bool ret = ImGui::Checkbox(label, &b);
		*v = b ? 1u : 0u;
		return ret;
	}

	// misc ImGui functions with additional item width parameter; extend when required

#define _Args(...) __VA_ARGS__
#define STRIP_PARENS(X) X
#define PASS_PARAMETERS(X) STRIP_PARENS( _Args X )
#define DEF_WITH_ITEMWIDTH(func_,declparams_,callparams_)		\
	bool func_ ## W(const float itemwidth, ## declparams_) {	\
		ImGui::PushItemWidth(itemwidth);						\
		bool ret = ImGui:: ## func_ ## callparams_;				\
		ImGui::PopItemWidth();									\
		return ret;												\
	}

DEF_WITH_ITEMWIDTH(Combo,	PASS_PARAMETERS((const char* label, int* current_item, const char* const items[], int items_count, int popup_max_height_in_items = -1)),	(label, current_item, items, items_count, popup_max_height_in_items))
DEF_WITH_ITEMWIDTH(Combo,	PASS_PARAMETERS((const char* label, int* current_item, const char* items_separated_by_zeros, int popup_max_height_in_items = -1)),			(label, current_item, items_separated_by_zeros, popup_max_height_in_items))
DEF_WITH_ITEMWIDTH(Combo,	PASS_PARAMETERS((const char* label, int* current_item, bool(*items_getter)(void* data, int idx, const char** out_text), void* data, int items_count, int popup_max_height_in_items = -1)),	(label, current_item, items_getter, data, items_count, popup_max_height_in_items))

DEF_WITH_ITEMWIDTH(InputFloat,	PASS_PARAMETERS((const char* label, float* v, float step = 0.0f, float step_fast = 0.0f, const char* format = "%.3f", ImGuiInputTextFlags flags = 0)),	(label, v, step, step_fast, format, flags))
DEF_WITH_ITEMWIDTH(InputFloat2,	PASS_PARAMETERS((const char* label, float v[2], const char* format = "%.3f", ImGuiInputTextFlags flags = 0)),											(label, v, format, flags))
DEF_WITH_ITEMWIDTH(InputFloat3,	PASS_PARAMETERS((const char* label, float v[3], const char* format = "%.3f", ImGuiInputTextFlags flags = 0)),											(label, v, format, flags))
DEF_WITH_ITEMWIDTH(InputFloat4,	PASS_PARAMETERS((const char* label, float v[4], const char* format = "%.3f", ImGuiInputTextFlags flags = 0)),											(label, v, format, flags))

DEF_WITH_ITEMWIDTH(InputInt,	PASS_PARAMETERS((const char* label, int* v, int step = 1, int step_fast = 100, ImGuiInputTextFlags flags = 0)),											(label, v, step, step_fast, flags))
DEF_WITH_ITEMWIDTH(InputInt2,	PASS_PARAMETERS((const char* label, int v[2], ImGuiInputTextFlags flags = 0)),																			(label, v, flags))
DEF_WITH_ITEMWIDTH(InputInt3,	PASS_PARAMETERS((const char* label, int v[3], ImGuiInputTextFlags flags = 0)),																			(label, v, flags))
DEF_WITH_ITEMWIDTH(InputInt4,	PASS_PARAMETERS((const char* label, int v[4], ImGuiInputTextFlags flags = 0)),																			(label, v, flags))

DEF_WITH_ITEMWIDTH(InputText,	PASS_PARAMETERS((const char* label, char* buf, size_t buf_size, ImGuiInputTextFlags flags = 0, ImGuiInputTextCallback callback = NULL, void* user_data = NULL)),	(label, buf, buf_size, flags, callback, user_data))
//DEF_WITH_ITEMWIDTH(SliderFloat,	PASS_PARAMETERS((const char* label, float* v, float v_min, float v_max, const char* format = "%.3f", float power = 1.0f)),											(label, v, v_min, v_max, format, power))
DEF_WITH_ITEMWIDTH(SliderFloat,	PASS_PARAMETERS((const char* label, float* v, float v_min, float v_max, const char* format = "%.3f", ImGuiSliderFlags flags = 0)),									(label, v, v_min, v_max, format, flags))
DEF_WITH_ITEMWIDTH(SliderInt,	PASS_PARAMETERS((const char* label, int* v, int v_min, int v_max, const char* format = "%d")),																		(label, v, v_min, v_max, format))

#undef DEF_WITH_ITEMWIDTH
#undef PASS_PARAMETERS
#undef STRIP_PARENS
#undef _Args

}
