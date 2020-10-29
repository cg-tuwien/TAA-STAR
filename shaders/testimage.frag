#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : enable
// -------------------------------------------------------

#include "shader_common_main.glsl"

layout (location = 0) in VertexData {
	vec2 texCoords;
} fs_in;

layout(set = 0, binding = 0) uniform sampler   uSampler;
layout(set = 0, binding = 1) uniform texture2D uInput;

layout(set = 1, binding = 0) UNIFORMDEF_MatricesAndUserInput uboMatUsr;

layout (location = 0) out vec4 oFragColor;

void main() {
#define JITTER_UV (params.mJitterNdcAndAlpha.xy * 0.5 * params.mUnjitterFactor)
	vec2 jitterUv = uboMatUsr.mJitterCurrentPrev.xy * 0.5 * -1.0; // jitter is in NDC units!
	vec2 uv = fs_in.texCoords + jitterUv;
	oFragColor = textureLod(sampler2D(uInput, uSampler), uv, 0);
}

