#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : enable
// -------------------------------------------------------

#include "shader_common_main.glsl"
#include "shader_cpu_common.h"

layout(set = 0, binding = 0) BUFFERDEF_Material materialsBuffer;
layout(set = 0, binding = 1) uniform sampler2D textures[];
layout(set = 1, binding = 0) UNIFORMDEF_MatricesAndUserInput uboMatUsr;

layout (location = 0) in VertexData
{
	vec2 texCoords;   // texture coordinates
	flat uint materialIndex;
} fs_in;
// -------------------------------------------------------

vec4 sample_from_diffuse_texture()
{
	uint matIndex = fs_in.materialIndex;
	int texIndex = materialsBuffer.materials[matIndex].mDiffuseTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mDiffuseTexOffsetTiling;
	vec2 texCoords = fs_in.texCoords * offsetTiling.zw + offsetTiling.xy;
	return SAMPLE_TEXTURE(textures[texIndex], texCoords);
}

void main()
{
	float alpha = sample_from_diffuse_texture().a;
	if (alpha < uboMatUsr.mUserInput.w) discard;
}

