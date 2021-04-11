#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : enable
// -------------------------------------------------------

#include "shader_common_main.glsl"
#include "shader_cpu_common.h"

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

// "mMatrices" uniform buffer containing camera matrices:
// It is updated every frame.
layout(set = 1, binding = 0) UNIFORMDEF_MatricesAndUserInput uboMatUsr;

// "mLightsources" uniform buffer containing all the light source data:
layout(set = 1, binding = 1) UNIFORMDEF_LightsourceData uboLights;

#if ENABLE_SHADOWMAP
layout(set = SHADOWMAP_BINDING_SET, binding = SHADOWMAP_BINDING_SLOT) uniform sampler2DShadow shadowMap[];
#endif

// -------------------------------------------------------

// include shadow calculation after uniforms are defined
#include "calc_shadows.glsl"

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
	return SAMPLE_TEXTURE(textures[texIndex], texCoords);
}

vec4 sample_from_specular_texture(int matIndex, vec2 uv)
{
	int texIndex = materialsBuffer.materials[matIndex].mSpecularTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mSpecularTexOffsetTiling;
	vec2 texCoords = uv * offsetTiling.zw + offsetTiling.xy;
	return SAMPLE_TEXTURE(textures[texIndex], texCoords);
}

vec4 sample_from_emissive_texture(int matIndex, vec2 uv) // ac
{
	int texIndex = materialsBuffer.materials[matIndex].mEmissiveTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mEmissiveTexOffsetTiling;
	vec2 texCoords = uv * offsetTiling.zw + offsetTiling.xy;
	return SAMPLE_TEXTURE(textures[texIndex], texCoords);
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

	#if ENABLE_SHADOWMAP
		vec4 worldSpace = inverse(uboMatUsr.mViewMatrix) * viewSpace;
		worldSpace /= worldSpace.w;
		float shadowFactor = calc_shadow_factor(worldSpace);
	#else
		float shadowFactor = 1.0;
	#endif

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

	// Calculate ambient illumination:
	vec3 ambientIllumination = vec3(0.0, 0.0, 0.0);
	for (uint i = uboLights.mRangesAmbientDirectional[0]; i < uboLights.mRangesAmbientDirectional[1]; ++i) {
		ambientIllumination += uboLights.mLightData[i].mColor.rgb * ambient;
	}

	// Calculate diffuse and specular illumination from all light sources:
	vec3 diffAndSpecIllumination = calc_illumination_in_vs(positionVS, normalVS, diff, spec, shininess);

	// Add all together:
	//vec4 blinnPhongColor = vec4(shadowFactor * vec3(ambientIllumination + emissive + diffAndSpecIllumination), 1.0);
	vec4 blinnPhongColor = vec4(vec3(ambientIllumination + emissive + shadowFactor * diffAndSpecIllumination), 1.0);


	if (uboMatUsr.mUserInput.z < 1.f) {
		oFragColor = blinnPhongColor;
	} else if (uboMatUsr.mUserInput.z < 2.f) {
		// don't use lights
		oFragColor = vec4(diff, 1.0);
	} else if (uboMatUsr.mUserInput.z < 3.f) {
		// Debug: SM cascade
		oFragColor = debug_shadow_cascade_color() * blinnPhongColor;
	} else {
		// other debug modes not used here
		oFragColor = vec4(1,0,1,1);
	}
}
// -------------------------------------------------------

