#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_post_depth_coverage : enable
// -------------------------------------------------------

#define TAU 6.28318530718 // TAU = 2 * PI

// ac: disable this, otherwise discarded fragments write to depth buffer!
//layout(early_fragment_tests) in;
//layout(post_depth_coverage) in;

// ###### MATERIAL DATA ##################################
// Material data struct definition:
struct MaterialGpuData {
	vec4 mDiffuseReflectivity;
	vec4 mAmbientReflectivity;
	vec4 mSpecularReflectivity;
	vec4 mEmissiveColor;
	vec4 mTransparentColor;
	vec4 mReflectiveColor;
	vec4 mAlbedo;

	float mOpacity;
	float mBumpScaling;
	float mShininess;
	float mShininessStrength;

	float mRefractionIndex;
	float mReflectivity;
	float mMetallic;
	float mSmoothness;

	float mSheen;
	float mThickness;
	float mRoughness;
	float mAnisotropy;

	vec4 mAnisotropyRotation;
	vec4 mCustomData;

	int mDiffuseTexIndex;
	int mSpecularTexIndex;
	int mAmbientTexIndex;
	int mEmissiveTexIndex;
	int mHeightTexIndex;
	int mNormalsTexIndex;
	int mShininessTexIndex;
	int mOpacityTexIndex;
	int mDisplacementTexIndex;
	int mReflectionTexIndex;
	int mLightmapTexIndex;
	int mExtraTexIndex;

	vec4 mDiffuseTexOffsetTiling;
	vec4 mSpecularTexOffsetTiling;
	vec4 mAmbientTexOffsetTiling;
	vec4 mEmissiveTexOffsetTiling;
	vec4 mHeightTexOffsetTiling;
	vec4 mNormalsTexOffsetTiling;
	vec4 mShininessTexOffsetTiling;
	vec4 mOpacityTexOffsetTiling;
	vec4 mDisplacementTexOffsetTiling;
	vec4 mReflectionTexOffsetTiling;
	vec4 mLightmapTexOffsetTiling;
	vec4 mExtraTexOffsetTiling;
};

// The actual material buffer (of type MaterialGpuData):
// It is bound to descriptor set at index 0 and
// within the descriptor set, to binding location 0
layout(set = 0, binding = 0) buffer Material
{
	MaterialGpuData materials[];
} materialsBuffer;

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
// It is updated every frame.
layout(set = 1, binding = 0) uniform MatricesAndUserInput {
	// view matrix as returned from quake_camera
	mat4 mViewMatrix;
	// projection matrix as returned from quake_camera
	mat4 mProjMatrix;
	// transformation matrix which tranforms to camera's position
	mat4 mCamPos;
	// x = tessellation factor, y = displacement strength, z = use lighting, w unused
	vec4 mUserInput;
} uboMatUsr;

struct LightsourceGpuData
{
	/** Color of the light source. */
	vec4 mColor;
	/** Direction of the light source. */
	vec4 mDirection;
	/** Position of the light source. */
	vec4 mPosition;
	/** Angles, where the individual elements contain the following data: [0] cosine of halve outer cone angle, [1] cosine of halve inner cone angle, [2] falloff, [3] unused */
	vec4 mAnglesFalloff;
	/* Light source attenuation, where the individual elements contain the following data: [0] constant attenuation factor, [1] linear attenuation factor, [2] quadratic attenuation factor, [3], unused */
	vec4 mAttenuation;
	/** General information about the light source, where the individual elements contain the following data:[0] type of the light source */
	ivec4 mInfo;
};

// "mLightsources" uniform buffer containing all the light source data:
layout(set = 1, binding = 1) uniform LightsourceData
{
	// x,y ... ambient light sources start and end indices; z,w ... directional light sources start and end indices
	uvec4 mRangesAmbientDirectional;
	// x,y ... point light sources start and end indices; z,w ... spot light sources start and end indices
	uvec4 mRangesPointSpot;
	// Contains all the data of all the active light sources
	LightsourceGpuData mLightData[128];
} uboLights;
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
	return texture(textures[texIndex], texCoords);
}

// ac:
vec4 sample_from_diffuse_texture()
{
	int matIndex = pushConstants.mMaterialIndex;
	int texIndex = materialsBuffer.materials[matIndex].mDiffuseTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mDiffuseTexOffsetTiling;
	vec2 texCoords = fs_in.texCoords * offsetTiling.zw + offsetTiling.xy;
	return texture(textures[texIndex], texCoords);
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
	// ac: discard transparent parts (hack to at least see transparency in the deferred shading setup - not really nice)
	//if (sample_from_diffuse_texture().a < 0.5) { discard; return; }

	vec3 normalVS = calc_normalized_normalVS(sample_from_normals_texture().rgb);
	float l = length(normalVS.xy);
	vec2 sphericalVS = vec2((l == 0) ? 0 : acos(clamp(normalVS.x / l, -1, 1)), asin(normalVS.z));
	if (normalVS.y < 0) sphericalVS.x = TAU - sphericalVS.x;
	oFragUvNrm = vec4(fs_in.texCoords, sphericalVS);
	oFragMatId = pushConstants.mMaterialIndex;
}
// -------------------------------------------------------

