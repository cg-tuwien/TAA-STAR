//? #version 460
//? #extension GL_EXT_nonuniform_qualifier : require
// all the // ? lines are just for the VS GLSL language integration plugin

#ifndef CALC_SHADOWS_INCLUDED
#define CALC_SHADOWS_INCLUDED 1

#include "shader_common_main.glsl"
#include "shader_cpu_common.h"

//? layout(set = 1, binding = 0) UNIFORMDEF_MatricesAndUserInput uboMatUsr;
#if ENABLE_SHADOWMAP
//? layout(set = SHADOWMAP_BINDING_SET, binding = SHADOWMAP_BINDING_SLOT) uniform sampler2DShadow shadowMap[];
#endif

int gDebugShadowCascade = -1;
float calc_shadow_factor(in vec4 positionWS) {
#if ENABLE_SHADOWMAP
	if (!uboMatUsr.mUseShadowMap) return 1.0;

	// find cascade to use // TODO - optimize
	int cascade = SHADOWMAP_NUM_CASCADES-1;
	for (int i = 0; i < SHADOWMAP_NUM_CASCADES-1; ++i) {
		if (gl_FragCoord.z < uboMatUsr.mShadowMapMaxDepth[i]) {
			cascade = i;
			break;
		}
	}
	gDebugShadowCascade = cascade;


	float light = 1.0;

	vec4 p = uboMatUsr.mShadowmapProjViewMatrix[cascade] * positionWS;
	p /= p.w; // no w-division should be needed if light proj is ortho... but .w could be off due to bone transforms (?), so do it anyway
	p.xy = p.xy * .5 + .5;
	if (all(greaterThanEqual(p.xyz, vec3(0))) && all(lessThan(p.xyz, vec3(1)))) {
		p.z -= uboMatUsr.mShadowBias;	// FIXME - using manual bias for now
		light = texture(shadowMap[cascade], p.xyz);
		light = 1.0 - (1.0 - light) * 0.75;
	}

	return light;
#else
	return 1.0;
#endif
}

vec4 debug_shadow_cascade_color() {
	vec3 cascCol;
	if      (gDebugShadowCascade <  0) cascCol = vec3(1);
	else if (gDebugShadowCascade == 0) cascCol = vec3(0, 1, 0);
	else if (gDebugShadowCascade == 1) cascCol = vec3(1, 1, 0);
	else if (gDebugShadowCascade == 2) cascCol = vec3(0.6, 0.6, 1);
	else                               cascCol = vec3(1, 0.6, 1);
	return vec4(cascCol,1);
}


#endif
