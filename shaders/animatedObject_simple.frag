#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_post_depth_coverage : enable
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

layout (location = 0) out vec4 oFragColor;
layout (location = 1) out uint oFragMatId;
layout (location = 2) out vec4 oFragVelocity;

void main()
{
	uint matIndex = fs_in.materialIndex;
	int texIndex = materialsBuffer.materials[matIndex].mDiffuseTexIndex;
	vec4 col = texture(textures[texIndex], fs_in.texCoords);

	oFragColor = vec4(col.rgb,1);
	oFragMatId = 0;
	oFragVelocity = vec4(0);
}

