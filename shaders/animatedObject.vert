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

layout(push_constant) uniform PushConstantsDII {
	mat4  mMover_modelMatrix;
	mat4  mMover_modelMatrix_prev;
	int   mMover_materialIndex;
	int   mMover_meshIndex;

	int   mDrawIdOffset; // negative numbers -> moving object id
	float pad1;
};


// "mMatrices" uniform buffer containing camera matrices:
// It is updated every frame.
layout(set = 1, binding = 0) UNIFORMDEF_MatricesAndUserInput uboMatUsr;

layout(set = 3, binding = 0, std430) readonly buffer BoneMatricesBuffer     { mat4 mat[]; } boneMatrices;
layout(set = 3, binding = 1, std430) readonly buffer BoneMatricesPrevBuffer { mat4 mat[]; } boneMatricesPrev;

// -------------------------------------------------------

// ###### DATA PASSED ON ALONG THE PIPELINE ##############
// Data from vert -> tesc or frag:
layout (location = 0) out VertexData {
	vec3 positionOS;
	vec3 positionVS;
	vec2 texCoords;
	vec3 normalOS;
	//vec3 tangentOS;
	//vec3 bitangentOS;
	vec4 positionCS;		// TODO: don't really need this!
	vec4 positionCS_prev;	// position in previous frame

	flat uint materialIndex;
	flat mat4 modelMatrix;
	flat int movingObjectId;
} v_out;
// -------------------------------------------------------

// ###### VERTEX SHADER MAIN #############################
void main()
{
	// moving object
	v_out.materialIndex = mMover_materialIndex;
	v_out.modelMatrix   = mMover_modelMatrix;
	v_out.movingObjectId = -mDrawIdOffset;

	// "normalize" bone weights - there may be more than four in the model, but we only get the first four here; make sure they add up to one
	vec4 boneWeights = aBoneWeights;
	boneWeights.w = 1.0 - boneWeights.x - boneWeights.y - boneWeights.z;

	uint bonesBaseIndex = mMover_meshIndex * MAX_BONES;

	// weighted sum of the four bone matrices
	mat4 boneMat =      boneMatrices.mat    [bonesBaseIndex + aBoneIndices[0]] * boneWeights[0]
				      + boneMatrices.mat    [bonesBaseIndex + aBoneIndices[1]] * boneWeights[1]
				      + boneMatrices.mat    [bonesBaseIndex + aBoneIndices[2]] * boneWeights[2]
				      + boneMatrices.mat    [bonesBaseIndex + aBoneIndices[3]] * boneWeights[3];

	mat4 prev_boneMat = boneMatricesPrev.mat[bonesBaseIndex + aBoneIndices[0]] * boneWeights[0]
					  + boneMatricesPrev.mat[bonesBaseIndex + aBoneIndices[1]] * boneWeights[1]
					  + boneMatricesPrev.mat[bonesBaseIndex + aBoneIndices[2]] * boneWeights[2]
					  + boneMatricesPrev.mat[bonesBaseIndex + aBoneIndices[3]] * boneWeights[3];

	mat4 mMatrix = v_out.modelMatrix;
	mat4 vMatrix = uboMatUsr.mViewMatrix;
	mat4 pMatrix = uboMatUsr.mProjMatrix;
	mat4 vmMatrix = vMatrix * mMatrix;
	mat4 pvmMatrix = pMatrix * vmMatrix;

	vec4 positionOS  = boneMat * vec4(aPosition, 1.0);
	vec4 positionVS  = vmMatrix * positionOS;
	vec4 positionCS  = pMatrix * positionVS;

	vec3 normalOS = normalize(mat3(boneMat) * normalize(aNormal));

	//vec3 tangentOS   = normalize(aTangent);
	//vec3 bitangentOS = normalize(aBitangent);

	mat4 prev_modelMatrix = mMover_modelMatrix_prev;

	v_out.positionOS  = positionOS.xyz;
	v_out.positionVS  = positionVS.xyz;
	v_out.texCoords   = aTexCoords;
	v_out.normalOS    = normalOS;
	//v_out.tangentOS   = tangentOS;
	//v_out.bitangentOS = bitangentOS;
	v_out.positionCS  = positionCS;	// TODO: recheck - is it ok to interpolate clip space vars?
	v_out.positionCS_prev = uboMatUsr.mPrevFrameProjViewMatrix * prev_modelMatrix * prev_boneMat * vec4(aPosition, 1.0);

	gl_Position = positionCS;
}
// -------------------------------------------------------

