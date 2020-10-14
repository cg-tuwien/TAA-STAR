#pragma once

// provide access to Renderdoc API for capturing, object labeling, etc.

#define USE_RENDERDOC_API 1





// Note: for now this module is agnostic of gears-vk, auto-vk and therefore uses simple printf for debug output

//#include <vulkan/vulkan.hpp>
#include <vector>

#if USE_RENDERDOC_API

// only include if we really want to use Renderdoc, so compilation is possible without having renderdoc_app.h
#include "renderdoc_app.h"

namespace rdoc {

	RENDERDOC_API_1_1_2* rdoc_api = nullptr;
	PFN_vkDebugMarkerSetObjectNameEXT pfnDebugMarkerSetObjectNameEXT = VK_NULL_HANDLE;
	VkDevice currentDevice = VK_NULL_HANDLE;

	void init() {
		if (HMODULE mod = GetModuleHandleA("renderdoc.dll")) {
			pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
			int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void **)&rdoc_api);
			assert(ret == 1);
			printf("RenderDoc found, API enabled\n");
		}
	}

	void init_debugmarkers(VkDevice device) {
		if (!rdoc_api) return;
		if (pfnDebugMarkerSetObjectNameEXT) return;

		currentDevice = device; // store, so we don't need to pass it in every time

		pfnDebugMarkerSetObjectNameEXT = reinterpret_cast<PFN_vkDebugMarkerSetObjectNameEXT>(vkGetDeviceProcAddr(device, "vkDebugMarkerSetObjectNameEXT"));
		printf(pfnDebugMarkerSetObjectNameEXT ? "Debugmarkers inited\n" : "!!! Failed to init debugmarkers; did you request VK_EXT_debug_marker ?\n");
	}

	bool active()			{ return rdoc_api; }
	void start_capture()	{ if (rdoc_api) rdoc_api->StartFrameCapture(NULL, NULL); }
	void end_capture()		{ if (rdoc_api) rdoc_api->EndFrameCapture(NULL, NULL); }

	void labelObject(uint64_t object, VkDebugReportObjectTypeEXT objectType, const char *objectName, int optionalIndex = -1) {
		if (!pfnDebugMarkerSetObjectNameEXT) return;
		VkDebugMarkerObjectNameInfoEXT nameInfo = {};
		nameInfo.sType			= VK_STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_NAME_INFO_EXT;
		nameInfo.objectType		= objectType;
		nameInfo.object			= object;
		if (optionalIndex < 0) {
			nameInfo.pObjectName = objectName;
			pfnDebugMarkerSetObjectNameEXT(currentDevice, &nameInfo);
		} else {
			std::string s = std::string(objectName) + "[" + std::to_string(optionalIndex) + "]";
			nameInfo.pObjectName = s.c_str();
			pfnDebugMarkerSetObjectNameEXT(currentDevice, &nameInfo);	// call this while s is still alive
		}
	}

	void labelImage(VkImage image, const char *name, int optionalIndex = -1) { labelObject(uint64_t(image), VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT, name, optionalIndex); }

	std::vector<const char*> required_device_extensions() { if (rdoc_api) return { "VK_EXT_debug_marker" }; else return {}; }
}

#else

namespace rdoc {
	void init() {}
	void init_debugmarkers(VkDevice device) {}
	bool active()			{ return false; }
	void start_capture()	{}
	void end_capture()		{}
	void labelObject(uint64_t object, VkDebugReportObjectTypeEXT objectType, const char *objectName, int optionalIndex = -1) {}
	void labelImage(VkImage image, const char *name, int optionalIndex = -1) {}
	std::vector<const char*> required_device_extensions() { return {}; }
}

#endif



