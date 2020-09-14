#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : enable
// -------------------------------------------------------

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
layout(push_constant) uniform PushConstants {
	mat4 mModelMatrix;
	int mMaterialIndex;
} pushConstants;
// "mMatrices" uniform buffer containing camera matrices:
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
	vec2 texCoords;   // texture coordinates
} fs_in;
layout (input_attachment_index = 0, set = 2, binding = 0) uniform subpassInput iDepth;
layout (input_attachment_index = 1, set = 2, binding = 1) uniform subpassInput iUvNrm;
layout (input_attachment_index = 2, set = 2, binding = 2) uniform usubpassInput iMatId;
// -------------------------------------------------------

// ###### FRAG OUTPUT ####################################
layout (location = 0) out vec4 oFragColor;
// -------------------------------------------------------

// ###### HELPER FUNCTIONS ###############################
vec4 sample_from_diffuse_texture(int matIndex, vec2 uv)
{
	int texIndex = materialsBuffer.materials[matIndex].mDiffuseTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mDiffuseTexOffsetTiling;
	vec2 texCoords = uv * offsetTiling.zw + offsetTiling.xy;
	return texture(textures[texIndex], texCoords);
}

vec4 sample_from_specular_texture(int matIndex, vec2 uv)
{
	int texIndex = materialsBuffer.materials[matIndex].mSpecularTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mSpecularTexOffsetTiling;
	vec2 texCoords = uv * offsetTiling.zw + offsetTiling.xy;
	return texture(textures[texIndex], texCoords);
}

vec4 sample_from_emissive_texture(int matIndex, vec2 uv) // ac
{
	int texIndex = materialsBuffer.materials[matIndex].mEmissiveTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mEmissiveTexOffsetTiling;
	vec2 texCoords = uv * offsetTiling.zw + offsetTiling.xy;
	return texture(textures[texIndex], texCoords);
}

// Calculates the light attenuation dividend for the given attenuation vector.
// @param atten attenuation data
// @param dist  distance
// @param dist2 squared distance
float calc_attenuation(vec4 atten, float dist, float dist2)
{
	return atten[0] + atten[1] * dist + atten[2] * dist2;
}

// Calculates the diffuse and specular illumination contribution for the given
// parameters according to the Blinn-Phong lighting model.
// All parameters must be normalized.
vec3 calc_blinn_phong_contribution(vec3 toLight, vec3 toEye, vec3 normal, vec3 diffFactor, vec3 specFactor, float specShininess)
{
	float nDotL = max(0.0, dot(normal, toLight)); // lambertian coefficient
	vec3 h = normalize(toLight + toEye);
	float nDotH = max(0.0, dot(normal, h));
	float specPower = (nDotH == 0 && specShininess == 0) ? 1 : pow(nDotH, specShininess);

	vec3 diffuse = diffFactor * nDotL; // component-wise product
	vec3 specular = specFactor * specPower;

	return diffuse + specular;
}

// Calculates the diffuse and specular illumination contribution for all the light sources.
// All calculations are performed in view space
vec3 calc_illumination_in_vs(vec3 posVS, vec3 normalVS, vec3 diff, vec3 spec, float shini)
{
	vec3 diffAndSpec = vec3(0.0, 0.0, 0.0);

	// Calculate shading in view space since all light parameters are passed to the shader in view space
	vec3 eyePosVS = vec3(0.0, 0.0, 0.0);
	vec3 toEyeNrmVS = normalize(eyePosVS - posVS);

	// directional lights
	for (uint i = uboLights.mRangesAmbientDirectional[2]; i < uboLights.mRangesAmbientDirectional[3]; ++i) {
		vec3 toLightDirVS = normalize(-uboLights.mLightData[i].mDirection.xyz);
		vec3 dirLightIntensity = uboLights.mLightData[i].mColor.rgb;
		diffAndSpec += dirLightIntensity * calc_blinn_phong_contribution(toLightDirVS, toEyeNrmVS, normalVS, diff, spec, shini);
	}

	// point lights
	for (uint i = uboLights.mRangesPointSpot[0]; i < uboLights.mRangesPointSpot[1]; ++i)
	{
		vec3 lightPosVS = uboLights.mLightData[i].mPosition.xyz;
		vec3 toLight = lightPosVS - posVS;
		float distSq = dot(toLight, toLight);
		float dist = sqrt(distSq);
		vec3 toLightNrm = toLight / dist;

		float atten = calc_attenuation(uboLights.mLightData[i].mAttenuation, dist, distSq);
		vec3 intensity = uboLights.mLightData[i].mColor.rgb / atten;

		diffAndSpec += intensity * calc_blinn_phong_contribution(toLightNrm, toEyeNrmVS, normalVS, diff, spec, shini);
	}

	// spot lights
	for (uint i = uboLights.mRangesPointSpot[2]; i < uboLights.mRangesPointSpot[3]; ++i)
	{
		vec3 lightPosVS = uboLights.mLightData[i].mPosition.xyz;
		vec3 toLight = lightPosVS - posVS;
		float distSq = dot(toLight, toLight);
		float dist = sqrt(distSq);
		vec3 toLightNrm = toLight / dist;

		float atten = calc_attenuation(uboLights.mLightData[i].mAttenuation, dist, distSq);
		vec3 intensity = uboLights.mLightData[i].mColor.rgb / atten;

		vec3 dirVS = uboLights.mLightData[i].mDirection.xyz;
		float cosOfHalfOuter = uboLights.mLightData[i].mAnglesFalloff[0];
		float cosOfHalfInner = uboLights.mLightData[i].mAnglesFalloff[1];
		float falloff = uboLights.mLightData[i].mAnglesFalloff[2];
		float cosAlpha = dot(-toLightNrm, dirVS);
		float da = cosAlpha - cosOfHalfOuter;
		float fade = cosOfHalfInner - cosOfHalfOuter;
		intensity *= da <= 0.0 ? 0.0 : pow(min(1.0, da / max(0.0001, fade)), falloff);

		diffAndSpec += intensity * calc_blinn_phong_contribution(toLightNrm, toEyeNrmVS, normalVS, diff, spec, shini);
	}

	return diffAndSpec;
}
// -------------------------------------------------------

// ###### VERTEX SHADER MAIN #############################
void main()
{
	float   depth =     subpassLoad(iDepth).r;
	vec4 uvNormal =     subpassLoad(iUvNrm).rgba;
	int  matIndex = int(subpassLoad(iMatId).r);

	if (depth == 1) discard;

	// unpack uv and normal
	vec2 uv = uvNormal.rg;
	vec3 normalVS = vec3(cos(uvNormal.z) * cos(uvNormal.w), sin(uvNormal.z) * cos(uvNormal.w), sin(uvNormal.w));

	// reconstruct position from depth buffer
	vec4 clipSpace = vec4(fs_in.texCoords * 2 - 1, depth, 1);
	vec4 viewSpace = inverse(uboMatUsr.mProjMatrix) * clipSpace;
	vec3 positionVS = viewSpace.xyz / viewSpace.w;

	vec3  diffTexColor = sample_from_diffuse_texture(matIndex, uv).rgb;
	float specTexValue = sample_from_specular_texture(matIndex, uv).r;
	vec3  emissiveTexColor = sample_from_emissive_texture(matIndex, uv).rgb; // ac

	// ac: hack, because emissiveTexColor is white now, if there is no texture assigned
	//if (emissiveTexColor == vec3(1)) emissiveTexColor = vec3(0);

	// Initialize all the colors:
	vec3 ambient    = materialsBuffer.materials[matIndex].mAmbientReflectivity.rgb * diffTexColor;
	//vec3 emissive   = materialsBuffer.materials[matIndex].mEmissiveColor.rgb;
	vec3 emissive   = materialsBuffer.materials[matIndex].mEmissiveColor.rgb * emissiveTexColor; // ac
	//vec3 emissive   = emissiveTexColor; // ac
	vec3 diff       = materialsBuffer.materials[matIndex].mDiffuseReflectivity.rgb * diffTexColor;
	vec3 spec       = materialsBuffer.materials[matIndex].mSpecularReflectivity.rgb * specTexValue;
	float shininess = materialsBuffer.materials[matIndex].mShininess;

	if (uboMatUsr.mUserInput.z != 0) {
		// Calculate ambient illumination:
		vec3 ambientIllumination = vec3(0.0, 0.0, 0.0);
		for (uint i = uboLights.mRangesAmbientDirectional[0]; i < uboLights.mRangesAmbientDirectional[1]; ++i) {
			ambientIllumination += uboLights.mLightData[i].mColor.rgb * ambient;
		}

		// Calculate diffuse and specular illumination from all light sources:
		vec3 diffAndSpecIllumination = calc_illumination_in_vs(positionVS, normalVS, diff, spec, shininess);

		// Add all together:
		oFragColor = vec4(ambientIllumination + emissive + diffAndSpecIllumination, 1.0);


	} else {
		//oFragColor = vec4(/* diff + */ emissive, 1.0);
		oFragColor = vec4(diff, 1.0);

		//oFragColor = vec4(normalVS, 1.0);

		//float alpha = sample_from_diffuse_texture(matIndex, uv).a;
		//oFragColor = vec4(diffTexColor, alpha);
		//if (alpha < 1.0) oFragColor = vec4(1,0,1,1);
	}
}
// -------------------------------------------------------
