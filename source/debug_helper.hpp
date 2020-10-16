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
	PFN_vkDebugMarkerSetObjectNameEXT	pfnDebugMarkerSetObjectNameEXT	= VK_NULL_HANDLE;
	PFN_vkCmdDebugMarkerBeginEXT		pfnCmdDebugMarkerBeginEXT		= VK_NULL_HANDLE;
	PFN_vkCmdDebugMarkerEndEXT			pfnCmdDebugMarkerEndEXT			= VK_NULL_HANDLE;
	VkDevice currentDevice = VK_NULL_HANDLE;
	bool capturingActive = false;

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

		pfnDebugMarkerSetObjectNameEXT	= reinterpret_cast<PFN_vkDebugMarkerSetObjectNameEXT>	(vkGetDeviceProcAddr(device, "vkDebugMarkerSetObjectNameEXT"));
		pfnCmdDebugMarkerBeginEXT		= reinterpret_cast<PFN_vkCmdDebugMarkerBeginEXT>		(vkGetDeviceProcAddr(device, "vkCmdDebugMarkerBeginEXT"));
		pfnCmdDebugMarkerEndEXT			= reinterpret_cast<PFN_vkCmdDebugMarkerEndEXT>			(vkGetDeviceProcAddr(device, "vkCmdDebugMarkerEndEXT"));
		if (!pfnDebugMarkerSetObjectNameEXT || !pfnCmdDebugMarkerBeginEXT || !pfnCmdDebugMarkerEndEXT)
			printf("WARN: Failed to init debugmarkers; did you request device extension \"VK_EXT_debug_marker\" ?\n");
	}

	bool active()			{ return rdoc_api; }
	void start_capture()	{ if (rdoc_api && !capturingActive) { rdoc_api->StartFrameCapture(NULL, NULL); capturingActive = true;  } }
	void end_capture()		{ if (rdoc_api &&  capturingActive) { rdoc_api->EndFrameCapture  (NULL, NULL); capturingActive = false; } }

	void labelObject(uint64_t object, VkDebugReportObjectTypeEXT objectType, const char *objectName, int64_t optionalIndex = -1) {
		if (!pfnDebugMarkerSetObjectNameEXT) return;
		VkDebugMarkerObjectNameInfoEXT nameInfo = { VK_STRUCTURE_TYPE_DEBUG_MARKER_OBJECT_NAME_INFO_EXT };
		nameInfo.objectType	= objectType;
		nameInfo.object		= object;
		if (optionalIndex < 0) {
			nameInfo.pObjectName = objectName;
			pfnDebugMarkerSetObjectNameEXT(currentDevice, &nameInfo);
		} else {
			std::string s = std::string(objectName) + "[" + std::to_string(optionalIndex) + "]";
			nameInfo.pObjectName = s.c_str();
			pfnDebugMarkerSetObjectNameEXT(currentDevice, &nameInfo);	// call this while s is still alive
		}
	}

	void labelImage (VkImage image,   const char *name, int64_t optionalIndex = -1) { labelObject(uint64_t(image),  VK_DEBUG_REPORT_OBJECT_TYPE_IMAGE_EXT,  name, optionalIndex); }
	void labelBuffer(VkBuffer buffer, const char *name, int64_t optionalIndex = -1) { labelObject(uint64_t(buffer), VK_DEBUG_REPORT_OBJECT_TYPE_BUFFER_EXT, name, optionalIndex); }
	void beginSection(VkCommandBuffer cmd, const char *name, int64_t optionalIndex = -1) {
		if (!pfnCmdDebugMarkerBeginEXT || !pfnCmdDebugMarkerEndEXT) return;
		VkDebugMarkerMarkerInfoEXT markerInfo = { VK_STRUCTURE_TYPE_DEBUG_MARKER_MARKER_INFO_EXT };
		if (optionalIndex < 0) {
			markerInfo.pMarkerName = name;
			pfnCmdDebugMarkerBeginEXT(cmd, &markerInfo);
		} else {
			std::string s = std::string(name) + "[" + std::to_string(optionalIndex) + "]";
			markerInfo.pMarkerName = s.c_str();
			pfnCmdDebugMarkerBeginEXT(cmd, &markerInfo);	// call this while s is still alive
		}
	}
	void endSection(VkCommandBuffer cmd) {
		if (!pfnCmdDebugMarkerBeginEXT || !pfnCmdDebugMarkerEndEXT) return;
		pfnCmdDebugMarkerEndEXT(cmd);
	}

	std::vector<const char*> required_device_extensions() { if (rdoc_api) return { "VK_EXT_debug_marker" }; else return {}; }
}

#else

namespace rdoc {
	void init() {}
	void init_debugmarkers(VkDevice device) {}
	bool active()			{ return false; }
	void start_capture()	{}
	void end_capture()		{}
	void labelObject(uint64_t object, VkDebugReportObjectTypeEXT objectType, const char *objectName, int64_t optionalIndex = -1) {}
	void labelImage(VkImage image, const char *name, int64_t optionalIndex = -1) {}
	void labelBuffer(VkBuffer buffer, const char *name, int64_t optionalIndex = -1) {}
	void beginSection(VkCommandBuffer cmd, const char *name, int64_t optionalIndex = -1) {}
	void endSection(VkCommandBuffer cmd) {}
	std::vector<const char*> required_device_extensions() { return {}; }
}

#endif



