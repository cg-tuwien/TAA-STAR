#version 460
#extension GL_GOOGLE_include_directive : enable
// -------------------------------------------------------

#include "shader_common_main.glsl"

// no vertex input
// shader is called with vertex count 12 * 2; to be rendered as line list (light camera only), or
//                                    24 * 2  for light camera + debug camera


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

	int vertId = gl_VertexIndex;
	mat4 invPV;
	if (vertId < 24) {
		invPV = inverse(uboMatUsr.mShadowmapProjViewMatrix);
		out_color = vec3(1,1,0);
	} else {
		vertId -= 24;
		invPV = inverse(uboMatUsr.mDebugCamProjViewMatrix);
		out_color = vec3(0,1,0);
	}

	int lineId  = vertId / 2;
	int pointId = vertId % 2;
	vec3 ndc = coord[lines[lineId][pointId]];

	vec4 world = invPV * vec4(ndc, 1.);
	gl_Position = uboMatUsr.mProjMatrix * uboMatUsr.mViewMatrix * world;
}

