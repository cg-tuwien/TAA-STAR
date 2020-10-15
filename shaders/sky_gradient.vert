#version 460
#extension GL_GOOGLE_include_directive : enable
// -------------------------------------------------------

#include "shader_common_main.glsl"

layout (location = 0) in vec3 aVertexPosition;

layout (location = 0) out vec3 vSphereCoords;

layout(set = 0, binding = 0) UNIFORMDEF_MatricesAndUserInput uboMatUsr;


void main()
{
	vSphereCoords = aVertexPosition.xyz;
	vec4 position_cs = uboMatUsr.mProjMatrix * uboMatUsr.mViewMatrix * uboMatUsr.mCamPos * vec4(aVertexPosition, 1.0);
	gl_Position = position_cs;
}

