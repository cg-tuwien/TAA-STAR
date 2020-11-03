#version 460
#extension GL_GOOGLE_include_directive : enable
// -------------------------------------------------------

#include "shader_common_main.glsl"

layout (location = 0) in vec4 aPositionAndId;

layout(set = 1, binding = 0) UNIFORMDEF_MatricesAndUserInput uboMatUsr;

layout (location = 0) out flat float controlPointId;

void main() {
	gl_Position = uboMatUsr.mProjMatrix * uboMatUsr.mViewMatrix * vec4(aPositionAndId.xyz, 1.0);
	controlPointId = aPositionAndId.w;
	gl_PointSize = (controlPointId > 0) ? 10.0 : 5.0;
}
