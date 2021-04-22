//? #version 460
// above line is just for the VS GLSL language integration plugin

#ifndef SHADER_RAYTRACE_COMMON_INCLUDED
#define SHADER_RAYTRACE_COMMON_INCLUDED 1

void calc_ray(in vec2 pixelUV, in mat4 camTransform, float aspectRatio, out vec3 origin, out vec3 direction) {
	// pixelUV: target location in uv screen space (range [0,1])
	// aspectRatio = float(gl_LaunchSizeEXT.x) / float(gl_LaunchSizeEXT.y);

	vec2 d = pixelUV * 2.0 - 1.0;
    origin = vec3(0.0, 0.0, 0.0);
    //                                                 Up == +Y in World, but UV-coordinates have +Y pointing down
    //                                                   |    Forward == -Z n World Space
    //                                                   |      |
    //                                                   v      v 
    direction = normalize(vec3(d.x * aspectRatio, -d.y, -sqrt(3))); // 1 => sqrt(3) is the scaling factor from a fov of 90 to 60
	vec4 p1 = vec4(origin, 1.0);
	vec4 p2 = vec4(origin + direction, 1.0);
	vec4 vp1 = camTransform * p1;
	vec4 vp2 = camTransform * p2;
	origin = vec3(vp1);
	direction = vec3(normalize(vp2 - vp1));
}

// ----- uniform declarations

// Push constants for ray tracing
#define PUSHCONSTANTSDEF_RAYTRACING uniform PushConstantsRayTracing {												\
	mat4 mCameraTransform;																							\
	mat4 mCameraViewProjMatrix;																						\
    vec4 mLightDir;																									\
	vec4 mDirLightIntensity;																						\
	vec4 mAmbientLightIntensity;																					\
	float mNormalMappingStrength;																					\
	float mMaxRayLength;																							\
	int  mNumSamples;																								\
	int  mAnimObjFirstMeshId;																						\
	int  mAnimObjNumMeshes;																							\
	uint mDoShadows;		/* bit 0: general shadows, bit 1: shadows of transp. objs */							\
	bool mAugmentTAA;																								\
	bool mAugmentTAADebug;																							\
	bool mApproximateLod;																							\
	float pad1, pad2, pad3;																							\
}

// ----- uniform structure definitions

struct MainRayPayload {
	vec4 color; // only rgb used
	vec2 pixelCenterUV;
};

#endif
