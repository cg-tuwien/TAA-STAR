#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_post_depth_coverage : enable
// -------------------------------------------------------

#include "shader_common_main.glsl"

#define TAU 6.28318530718 // TAU = 2 * PI

// ac: disable this, otherwise discarded fragments write to depth buffer!
//layout(early_fragment_tests) in;
//layout(post_depth_coverage) in;

// ###### MATERIAL DATA ##################################
// The actual material buffer (of type MaterialGpuData):
// It is bound to descriptor set at index 0 and
// within the descriptor set, to binding location 0
layout(set = 0, binding = 0) BUFFERDEF_Material materialsBuffer;

// Array of samplers containing all the material's images:
// These samplers are referenced from materials by
// index, namely by all those m*TexIndex members.
layout(set = 0, binding = 1) uniform sampler2D textures[];
// -------------------------------------------------------

// ###### PIPELINE INPUT DATA ############################
// Unique push constants per draw call (You can think of
// these like single uniforms in OpenGL):
layout(push_constant) uniform PushConstants {
	mat4 mModelMatrix;
	int mMaterialIndex;
} pushConstants;

// Uniform buffer containing camera matrices and user input:
layout(set = 1, binding = 0) UNIFORMDEF_MatricesAndUserInput uboMatUsr;

// "mLightsources" uniform buffer containing all the light source data:
layout(set = 1, binding = 1) UNIFORMDEF_LightsourceData uboLights;

// -------------------------------------------------------

// ###### FRAG INPUT #####################################
layout (location = 0) in VertexData
{
	vec3 positionOS;  // not used in this shader
	vec3 positionVS;  // interpolated vertex position in view-space
	vec2 texCoords;   // texture coordinates
	vec3 normalOS;    // interpolated vertex normal in object-space
	vec3 tangentOS;   // interpolated vertex tangent in object-space
	vec3 bitangentOS; // interpolated vertex bitangent in object-space
} fs_in;
// -------------------------------------------------------

// ###### FRAG OUTPUT ####################################
layout (location = 0) out vec4 oFragUvNrm;
layout (location = 1) out uint oFragMatId;
// -------------------------------------------------------

// ###### HELPER FUNCTIONS ###############################

vec4 sample_from_normals_texture()
{
	int matIndex = pushConstants.mMaterialIndex;
	int texIndex = materialsBuffer.materials[matIndex].mNormalsTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mNormalsTexOffsetTiling;
	vec2 texCoords = fs_in.texCoords * offsetTiling.zw + offsetTiling.xy;
	return SAMPLE_TEXTURE(textures[texIndex], texCoords);
}

// ac:
vec4 sample_from_diffuse_texture()
{
	int matIndex = pushConstants.mMaterialIndex;
	int texIndex = materialsBuffer.materials[matIndex].mDiffuseTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mDiffuseTexOffsetTiling;
	vec2 texCoords = fs_in.texCoords * offsetTiling.zw + offsetTiling.xy;
	return SAMPLE_TEXTURE(textures[texIndex], texCoords);
}


// Re-orthogonalizes the first vector w.r.t. the second vector (Gram-Schmidt process)
vec3 re_orthogonalize(vec3 first, vec3 second)
{
	return normalize(first - dot(first, second) * second);
}

// Calculates the normalized normal in view space by sampling the
// normal from the normal map and transforming it with the TBN-matrix.
vec3 calc_normalized_normalVS(vec3 sampledNormal)
{
	mat4 vmMatrix = uboMatUsr.mViewMatrix * pushConstants.mModelMatrix;
	mat3 vmNormalMatrix = mat3(inverse(transpose(vmMatrix)));

	// build the TBN matrix from the varyings
	vec3 normalOS = normalize(fs_in.normalOS);
	vec3 tangentOS = re_orthogonalize(fs_in.tangentOS, normalOS);
	vec3 bitangentOS = re_orthogonalize(fs_in.bitangentOS, normalOS);

	mat3 matrixTStoOS = inverse(transpose(mat3(tangentOS, bitangentOS, normalOS)));

	// sample the normal from the normal map and bring it into view space
	vec3 normalSample = normalize(sampledNormal * 2.0 - 1.0);

	int matIndex = pushConstants.mMaterialIndex;
	float normalMappingStrengthFactor = 1.0f - materialsBuffer.materials[matIndex].mCustomData[2];

	float userDefinedDisplacementStrength = uboMatUsr.mUserInput[1];
	normalSample.xy *= userDefinedDisplacementStrength * normalMappingStrengthFactor;

	vec3 normalVS = vmNormalMatrix * matrixTStoOS * normalSample;

	return normalize(normalVS);
}

// -------------------------------------------------------

// ###### VERTEX SHADER MAIN #############################
void main()
{
	vec3 normalVS = calc_normalized_normalVS(sample_from_normals_texture().rgb);
	float l = length(normalVS.xy);
	vec2 sphericalVS = vec2((l == 0) ? 0 : acos(clamp(normalVS.x / l, -1, 1)), asin(normalVS.z));
	if (normalVS.y < 0) sphericalVS.x = TAU - sphericalVS.x;
	oFragUvNrm = vec4(fs_in.texCoords, sphericalVS);
	oFragMatId = pushConstants.mMaterialIndex;
}
// -------------------------------------------------------

