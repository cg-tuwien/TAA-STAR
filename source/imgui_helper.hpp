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
}
