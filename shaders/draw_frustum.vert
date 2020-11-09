#version 460
#extension GL_GOOGLE_include_directive : enable
// -------------------------------------------------------

#include "shader_common_main.glsl"
#include "shader_cpu_common.h"

// no vertex input
// shader is called with vertex count 12 * 2 * 5; to be rendered as line list (4 light cameras + debug camera)


layout(set = 1, binding = 0) UNIFORMDEF_MatricesAndUserInput uboMatUsr;

layout (location = 0) out vec3 out_color;

void main() {
	const float zmin = 0;
	const float zmax = 1;
	vec3 coord[8] = {
	  {-1,-1,zmin}, {-1,1,zmin}, {1,1,zmin}, {1,-1,zmin},
	  {-1,-1,zmax}, {-1,1,zmax}, {1,1,zmax}, {1,-1,zmax}
	};
	ivec2 lines[12] = {
		{0,1}, {1,2}, {2,3}, {3,0},	// near plane
		{4,5}, {5,6}, {6,7}, {7,4},	// far plane
		{0,4}, {1,5}, {2,6}, {3,7}  // sides
	};
	vec3 cascCol[5] = {
		{0, 1, 0},
		{1, 1, 0},
		{0.6, 0.6, 1},
		{1, 0.6, 1},
		{1, 0, 1}
	};

	int vertId = gl_VertexIndex;
	int cascade = vertId / 24;
	vertId %= 24;

	if (cascade < 4 && cascade >= SHADOWMAP_NUM_CASCADES) {
		out_color = vec3(0);
		gl_Position = vec4(2,2,2,1);
		return;
	}

	mat4 invPV;
	if (cascade < 4) {
		invPV = inverse(uboMatUsr.mShadowmapProjViewMatrix[cascade]);
	} else {
		invPV = inverse(uboMatUsr.mDebugCamProjViewMatrix);
	}


	int lineId  = vertId / 2;
	int pointId = vertId % 2;
	vec3 ndc = coord[lines[lineId][pointId]];

	vec4 world = invPV * vec4(ndc, 1.);
	gl_Position = uboMatUsr.mProjMatrix * uboMatUsr.mViewMatrix * world;
	out_color = cascCol[cascade];
}

