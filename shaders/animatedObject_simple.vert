#version 460
#extension GL_GOOGLE_include_directive : enable
// -------------------------------------------------------

#include "shader_common_main.glsl"
#include "shader_cpu_common.h"

// TODO: fix tangents, bitangents


// ###### VERTEX SHADER/PIPELINE INPUT DATA ##############
// Several vertex attributes (These are the buffers passed
// to command_buffer_t::draw_indexed in the same order):
layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec2 aTexCoords;
layout (location = 2) in vec3 aNormal;
layout (location = 3) in vec4 aBoneWeights;
layout (location = 4) in uvec4 aBoneIndices;

layout(push_constant) uniform PushConstantsDII { int mDrawIdOffset; };	// negative: moving object


// "mMatrices" uniform buffer containing camera matrices:
// It is updated every frame.
layout(set = 1, binding = 0) UNIFORMDEF_MatricesAndUserInput uboMatUsr;

layout(set = 3, binding = 0, std430) readonly buffer BoneMatricesBuffer {
	mat4 mat[];
} boneMatrices;

// -------------------------------------------------------

// ###### DATA PASSED ON ALONG THE PIPELINE ##############
// Data from vert -> tesc or frag:
layout (location = 0) out VertexData {
	vec2 texCoords;
} v_out;
// -------------------------------------------------------

// ###### VERTEX SHADER MAIN #############################
void main()
{
	// "normalize" bone weights - there may be more than four in the model, but we only get the first four here; make sure they add up to one
	vec4 boneWeights = aBoneWeights;
	boneWeights.w = 1.0 - dot(aBoneWeights.xyz, vec3(1,1,1));

	vec4 posOS  = boneWeights[0] * (boneMatrices.mat[aBoneIndices[0]] * vec4(aPosition, 1))
				+ boneWeights[1] * (boneMatrices.mat[aBoneIndices[1]] * vec4(aPosition, 1))
				+ boneWeights[2] * (boneMatrices.mat[aBoneIndices[2]] * vec4(aPosition, 1))
				+ boneWeights[3] * (boneMatrices.mat[aBoneIndices[3]] * vec4(aPosition, 1));

	vec4 posWS = uboMatUsr.mMovingObjectModelMatrix * posOS;

	v_out.texCoords   = aTexCoords;
	gl_Position = uboMatUsr.mProjMatrix * uboMatUsr.mViewMatrix * posWS;
}
// -------------------------------------------------------

