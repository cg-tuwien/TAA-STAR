#version 460
#extension GL_GOOGLE_include_directive : enable
// -------------------------------------------------------

#include "shader_common_main.glsl"

layout (location = 0) in vec3 aPosition;

layout(set = 1, binding = 0) UNIFORMDEF_MatricesAndUserInput uboMatUsr;

void main() {
	gl_PointSize = 5.0;
	gl_Position = uboMatUsr.mProjMatrix * uboMatUsr.mViewMatrix * vec4(aPosition, 1.0);
}
