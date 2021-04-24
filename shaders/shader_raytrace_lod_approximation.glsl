//? #version 460
// above line is just for the VS GLSL language integration plugin
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : enable

#include "shader_cpu_common.h"
#include "shader_common_main.glsl"
#include "shader_raytrace_common.glsl"

// ###### LOD APPROXIMATION ########################

// interim results for lod calculation
bool gLodApprox_valid  = false;
vec2 gLodApprox_dx_vuv = vec2(0);
vec2 gLodApprox_dy_vuv = vec2(0);

//float approximate_lod_from_triangle_area(vec3 P0_OS, vec3 P1_OS, vec3 P2_OS, vec2 uv0, vec2 uv1, vec2 uv2, uint matIndex) {
//	// screen-space (pixels) triangle area
//	vec4 P0_CS = pushConstants.mCameraViewProjMatrix * vec4(gl_ObjectToWorldEXT * vec4(P0_OS,1), 1);
//	vec4 P1_CS = pushConstants.mCameraViewProjMatrix * vec4(gl_ObjectToWorldEXT * vec4(P1_OS,1), 1);
//	vec4 P2_CS = pushConstants.mCameraViewProjMatrix * vec4(gl_ObjectToWorldEXT * vec4(P2_OS,1), 1);
//	vec2 imSize = vec2(imageSize(image));
//	vec2 P0_SS = 0.5 * P0_CS.xy / P0_CS.w + 0.5;
//	vec2 P1_SS = 0.5 * P1_CS.xy / P1_CS.w + 0.5;
//	vec2 P2_SS = 0.5 * P2_CS.xy / P2_CS.w + 0.5;
//	//float doubleArea = length(cross((P1_SS - P0_SS), (P2_SS - P0_SS)));
//	vec2 v01 = P1_SS - P0_SS;
//	vec2 v02 = P2_SS - P0_SS;
//	float doubleAreaPixels = imSize.x * imSize.y * abs(v01.x * v02.y - v01.y * v02.x);
//
//	// texels triangle area
//	int texIndex = materialsBuffer.materials[matIndex].mDiffuseTexIndex;
//	vec2 texSize = textureSize(textures[texIndex], 0);
//	vec2 t01 = uv1 - uv0;
//	vec2 t02 = uv2 - uv0;
//	float doubleAreaTexels = texSize.x * texSize.y * abs(t01.x * t02.y - t01.y * t02.x);
//
//	return 0.5 * log2(doubleAreaTexels / doubleAreaPixels);
//}

// intersect_triangle adapted from https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d/42752998#42752998
// When the function returns true, the intersection point is given by Ray_Origin + t * Ray_Dir. The barycentric coordinates of the intersection in the triangle are u, v, 1-u-v
bool intersect_triangle(in vec3 Ray_Origin, in vec3 Ray_Dir, in vec3 A, in vec3 B, in vec3 C, out float t, out float u, out float v, out vec3 N) { 
   vec3 E1 = B-A;
   vec3 E2 = C-A;
         N = cross(E1,E2);
   float det = -dot(Ray_Dir, N);
   float invdet = 1.0/det;
   vec3 AO  = Ray_Origin - A;
   vec3 DAO = cross(AO, Ray_Dir);
   u =  dot(E2,DAO) * invdet;
   v = -dot(E1,DAO) * invdet;
   t =  dot(AO,N)   * invdet; 
   return (det >= 1e-6 && t >= 0.0 && u >= 0.0 && v >= 0.0 && (u+v) <= 1.0);
}

// this is just to avoid errors in the VS GLSL plugin:
//!mat4x3 dummyObjToWorldMat;
//!ivec2  dummyLaunchSize;
//!vec2   dummyPixelCenterUV;
//!mat4   dummyCameraTransform;
//!#define RAYTRACE_LOD_APPROXIMATION_OBJECT_TO_WORLD_MATRIX dummyObjToWorldMat
//!#define RAYTRACE_LOD_APPROXIMATION_LAUNCHSIZE dummyLaunchSize
//!#define RAYTRACE_LOD_APPROXIMATION_PIXELCENTERUV dummyPixelCenterUV
//!#define RAYTRACE_LOD_APPROXIMATION_CAMERATRANSFORM dummyCameraTransform

#ifndef RAYTRACE_LOD_APPROXIMATION_OBJECT_TO_WORLD_MATRIX
#define RAYTRACE_LOD_APPROXIMATION_OBJECT_TO_WORLD_MATRIX gl_ObjectToWorldEXT
#endif
#ifndef RAYTRACE_LOD_APPROXIMATION_LAUNCHSIZE
#define RAYTRACE_LOD_APPROXIMATION_LAUNCHSIZE gl_LaunchSizeEXT
#endif
#ifndef RAYTRACE_LOD_APPROXIMATION_PIXELCENTERUV
#define RAYTRACE_LOD_APPROXIMATION_PIXELCENTERUV hitValue.pixelCenterUV
#endif
#ifndef RAYTRACE_LOD_APPROXIMATION_CAMERATRANSFORM
#define RAYTRACE_LOD_APPROXIMATION_CAMERATRANSFORM pushConstants.mCameraTransform
#endif


void approximate_lod_homebrewed_setup(vec3 P0_OS, vec3 P1_OS, vec3 P2_OS, vec2 uv0, vec2 uv1, vec2 uv2, uint matIndex, vec2 uv_here) {
	// idea: calc triangle intersection of rays through (px+1,py) and through (px, py+1),
	//       fina uv coords, calc au/ax, av/ax, au/ay, av/ay

	vec3 P0_WS = RAYTRACE_LOD_APPROXIMATION_OBJECT_TO_WORLD_MATRIX * vec4(P0_OS,1);
	vec3 P1_WS = RAYTRACE_LOD_APPROXIMATION_OBJECT_TO_WORLD_MATRIX * vec4(P1_OS,1);
	vec3 P2_WS = RAYTRACE_LOD_APPROXIMATION_OBJECT_TO_WORLD_MATRIX * vec4(P2_OS,1);

	// calc ray origins and directions
	vec2 onePixel = vec2(1.0) / vec2(RAYTRACE_LOD_APPROXIMATION_LAUNCHSIZE);
	float aspectRatio = float(RAYTRACE_LOD_APPROXIMATION_LAUNCHSIZE.x) / float(RAYTRACE_LOD_APPROXIMATION_LAUNCHSIZE.y);
	vec3 origin_dx, origin_dy, direction_dx, direction_dy;
	calc_ray(RAYTRACE_LOD_APPROXIMATION_PIXELCENTERUV + vec2(onePixel.x, 0), RAYTRACE_LOD_APPROXIMATION_CAMERATRANSFORM, aspectRatio, origin_dx, direction_dx);
	calc_ray(RAYTRACE_LOD_APPROXIMATION_PIXELCENTERUV + vec2(0, onePixel.y), RAYTRACE_LOD_APPROXIMATION_CAMERATRANSFORM, aspectRatio, origin_dy, direction_dy);
	
	// find hit points (don't care if they actually fall into the triangle)
	vec2 bar_dx, bar_dy;
	float t;
	vec3 N;
	intersect_triangle(origin_dx, direction_dx, P0_WS, P1_WS, P2_WS, t, bar_dx.x, bar_dx.y, N);
	intersect_triangle(origin_dy, direction_dy, P0_WS, P1_WS, P2_WS, t, bar_dy.x, bar_dy.y, N);

	// get texture coordinates at hit points (attn: bary coord interpolation order)
	vec2 uv_dx = uv0 * (1.0 - bar_dx.x - bar_dx.y) + uv1 * bar_dx.x + uv2 * bar_dx.y;
	vec2 uv_dy = uv0 * (1.0 - bar_dy.x - bar_dy.y) + uv1 * bar_dy.x + uv2 * bar_dy.y;

	// result of this function: partial derivatives in uv space
	gLodApprox_dx_vuv = (uv_dx - uv_here);
	gLodApprox_dy_vuv = (uv_dy - uv_here);
	gLodApprox_valid  = true;
}

float approximate_lod_homebrewed_final(in vec2 texSize) {
	// finalize lod calculation for a particular texture

	// approximate_lod_homebrewed_setup() must have been called before, so that gLod_dx_vuv and gLod_dy_vuv are valid!
	// otherwise return level 0
	if (!gLodApprox_valid) return 0.0;

	// calc lod
	vec2 dx_vtc = texSize * gLodApprox_dx_vuv;
	vec2 dy_vtc = texSize * gLodApprox_dy_vuv;
	float delta_max_sqr = max(dot(dx_vtc, dx_vtc), dot(dy_vtc, dy_vtc));
	float lod = max(0, 0.5 * log2(delta_max_sqr));

	// aniso variant
	float px = dot(dx_vtc, dx_vtc);
	float py = dot(dy_vtc, dy_vtc);
	float maxLod = 0.5 * log2(max(px,py));
	float minLod = 0.5 * log2(min(px,py));
	const float maxAniso = 32;
	const float maxAnisoLog2 = log2(maxAniso);
	lod = maxLod - min(maxLod - minLod, maxAnisoLog2);

	return lod;
}

vec4 sampleTextureWithLodApprox(in sampler2D tex, vec2 uv) {
	float lod = approximate_lod_homebrewed_final(vec2(textureSize(tex, 0)));
	return textureLod(tex, uv, lod);
}
// -------------------------------------------------------
