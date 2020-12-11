#version 460
#extension GL_GOOGLE_include_directive : enable
// -------------------------------------------------------

#include "shader_common_main.glsl"
#include "shader_cpu_common.h"

// ###### VERTEX SHADER/PIPELINE INPUT DATA ##############
layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec4 aBoneWeights;
layout (location = 2) in uvec4 aBoneIndices;

layout(push_constant) PUSHCONSTANTSDEF_DII;
layout(set = 1, binding = 0) UNIFORMDEF_MatricesAndUserInput uboMatUsr;

layout(set = 3, binding = 0, std430) readonly buffer BoneMatricesBuffer     { mat4 mat[]; } boneMatrices;

// ###### VERTEX SHADER MAIN #############################
void main()
{
	mat4 modelMatrix     = uboMatUsr.mMover_additionalModelMatrix * mMover_baseModelMatrix;

	vec4 boneWeights = aBoneWeights;
	//boneWeights.w = 1.0 - boneWeights.x - boneWeights.y - boneWeights.z; // no longer necessary to "normalize", this is now done at model loading

	uint bonesBaseIndex = mMover_meshIndex * MAX_BONES;

	// weighted sum of the four bone matrices
	mat4 boneMat =      boneMatrices.mat    [bonesBaseIndex + aBoneIndices[0]] * boneWeights[0]
				      + boneMatrices.mat    [bonesBaseIndex + aBoneIndices[1]] * boneWeights[1]
				      + boneMatrices.mat    [bonesBaseIndex + aBoneIndices[2]] * boneWeights[2]
				      + boneMatrices.mat    [bonesBaseIndex + aBoneIndices[3]] * boneWeights[3];

	gl_Position = uboMatUsr.mShadowmapProjViewMatrix[mShadowMapCascadeToBuild] * modelMatrix * boneMat * vec4(aPosition, 1.0);
}
// -------------------------------------------------------

