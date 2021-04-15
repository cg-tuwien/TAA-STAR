#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_post_depth_coverage : enable
// -------------------------------------------------------

#include "shader_common_main.glsl"
#include "shader_cpu_common.h"

// ac: specialization constant to differentiate between opaque pass (0) and transparent pass (1)
layout(constant_id = SPECCONST_ID_TRANSPARENCY) const uint transparentPass = SPECCONST_VAL_OPAQUE;


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

// set 0, binding 2-4 used in vertex shader

#if ENABLE_SHADOWMAP
layout(set = SHADOWMAP_BINDING_SET, binding = SHADOWMAP_BINDING_SLOT) uniform sampler2DShadow shadowMap[];
#endif

// -------------------------------------------------------

// ###### PIPELINE INPUT DATA ############################

// Uniform buffer containing camera matrices and user input:
// It is updated every frame.
layout(set = 1, binding = 0) UNIFORMDEF_MatricesAndUserInput uboMatUsr;

// "mLightsources" uniform buffer containing all the light source data:
layout(set = 1, binding = 1) UNIFORMDEF_LightsourceData uboLights;

// -------------------------------------------------------

// include shadow calculation after uniforms are defined
#include "calc_shadows.glsl"

// -------------------------------------------------------

// ###### FRAG INPUT #####################################
layout (location = 0) in VertexData
{
	vec4 positionWS;  // pos world space
	vec3 positionVS;  // interpolated vertex position in view-space
	vec2 texCoords;   // texture coordinates
	vec3 normalOS;    // interpolated vertex normal in object-space
	vec3 tangentOS;   // interpolated vertex tangent in object-space
	vec3 bitangentOS; // interpolated vertex bitangent in object-space
	vec4 positionCS;		// TODO: don't really need this! can calc from gl_FragCoord
	vec4 positionCS_prev;	// position in previous frame

	flat uint materialIndex;
	flat mat4 modelMatrix;
	flat int movingObjectId;
} fs_in;
// -------------------------------------------------------

// ###### FRAG OUTPUT ####################################
layout (location = 0) out vec4 oFragColor;
layout (location = 1) out uint oFragMatId;
layout (location = 2) out vec4 oFragVelocity;
// -------------------------------------------------------

// ###### HELPER FUNCTIONS ###############################

vec4 sample_from_normals_texture()
{
	uint matIndex = fs_in.materialIndex;
	int texIndex = materialsBuffer.materials[matIndex].mNormalsTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mNormalsTexOffsetTiling;
	vec2 texCoords = fs_in.texCoords * offsetTiling.zw + offsetTiling.xy;
	vec4 normalSample = SAMPLE_TEXTURE(textures[texIndex], texCoords);
	FIX_NORMALMAPPING(normalSample);
	return normalSample;
}

vec4 sample_from_diffuse_texture()
{
	uint matIndex = fs_in.materialIndex;
	int texIndex = materialsBuffer.materials[matIndex].mDiffuseTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mDiffuseTexOffsetTiling;
	vec2 texCoords = fs_in.texCoords * offsetTiling.zw + offsetTiling.xy;
	return SAMPLE_TEXTURE(textures[texIndex], texCoords);
}

vec4 sample_from_specular_texture()
{
	uint matIndex = fs_in.materialIndex;
	int texIndex = materialsBuffer.materials[matIndex].mSpecularTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mSpecularTexOffsetTiling;
	vec2 texCoords = fs_in.texCoords * offsetTiling.zw + offsetTiling.xy;
	return SAMPLE_TEXTURE(textures[texIndex], texCoords);
}

vec4 sample_from_emissive_texture()
{
	uint matIndex = fs_in.materialIndex;
	int texIndex = materialsBuffer.materials[matIndex].mEmissiveTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mEmissiveTexOffsetTiling;
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
	mat4 vmMatrix = uboMatUsr.mViewMatrix * fs_in.modelMatrix;
	mat3 vmNormalMatrix = mat3(inverse(transpose(vmMatrix)));

	// build the TBN matrix from the varyings
	vec3 normalOS = normalize(fs_in.normalOS);
	vec3 tangentOS = re_orthogonalize(fs_in.tangentOS, normalOS);
	vec3 bitangentOS = re_orthogonalize(fs_in.bitangentOS, normalOS);

	mat3 matrixTStoOS = inverse(transpose(mat3(tangentOS, bitangentOS, normalOS)));

	// sample the normal from the normal map and bring it into view space
	vec3 normalSample = normalize(sampledNormal * 2.0 - 1.0);

	uint matIndex = fs_in.materialIndex;
	float normalMappingStrengthFactor = 1.0f - materialsBuffer.materials[matIndex].mCustomData[2];

	float userDefinedDisplacementStrength = uboMatUsr.mUserInput[1];
	normalSample.xy *= userDefinedDisplacementStrength * normalMappingStrengthFactor;

	vec3 normalVS = vmNormalMatrix * matrixTStoOS * normalSample;

	return normalize(normalVS);
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
vec3 calc_blinn_phong_contribution(vec3 toLight, vec3 toEye, vec3 normal, vec3 diffFactor, vec3 specFactor, float specShininess, bool twoSided)
{
	if (twoSided) {
		if (dot(normal, toEye) < 0) normal = -normal; // flip normal if it points away from us
	}

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
vec3 calc_illumination_in_vs(vec3 posVS, vec3 normalVS, vec3 diff, vec3 spec, float shini, bool twoSided)
{
	vec3 diffAndSpec = vec3(0.0, 0.0, 0.0);

	// Calculate shading in view space since all light parameters are passed to the shader in view space
	vec3 eyePosVS = vec3(0.0, 0.0, 0.0);
	vec3 toEyeNrmVS = normalize(eyePosVS - posVS);

	// directional lights
	for (uint i = uboLights.mRangesAmbientDirectional[2]; i < uboLights.mRangesAmbientDirectional[3]; ++i) {
		vec3 toLightDirVS = normalize(-uboLights.mLightData[i].mDirection.xyz);
		vec3 dirLightIntensity = uboLights.mLightData[i].mColor.rgb;
		diffAndSpec += dirLightIntensity * calc_blinn_phong_contribution(toLightDirVS, toEyeNrmVS, normalVS, diff, spec, shini, twoSided);
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

		diffAndSpec += intensity * calc_blinn_phong_contribution(toLightNrm, toEyeNrmVS, normalVS, diff, spec, shini, twoSided);
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

		diffAndSpec += intensity * calc_blinn_phong_contribution(toLightNrm, toEyeNrmVS, normalVS, diff, spec, shini, twoSided);
	}

	return diffAndSpec;
}

// -------------------------------------------------------

// ###### FRAGMENT SHADER MAIN #############################
void main()
{
	vec3 normalVS = calc_normalized_normalVS(sample_from_normals_texture().rgb);
	vec3 positionVS = fs_in.positionVS;
	uint matIndex = fs_in.materialIndex;

	vec4  diffTexColorRGBA = sample_from_diffuse_texture().rgba;
	float specTexValue     = sample_from_specular_texture().r;
	vec3  emissiveTexColor = sample_from_emissive_texture().rgb;

	float alpha = (transparentPass == SPECCONST_VAL_TRANSPARENT) ? diffTexColorRGBA.a : 1.0;

	// ac: ugly hack - discard very transparent parts ; this way we can get away without sorting and disabling depth_write, even when using alpha-blending
	if (transparentPass == SPECCONST_VAL_TRANSPARENT && alpha < uboMatUsr.mUserInput.w) { discard; return; }

	// write material // TODO: better flag specific/problematic materials that require different TAA handling
	oFragMatId = (fs_in.materialIndex + 1) | (fs_in.movingObjectId != 0 ? 0x80000000 : 0);

	// calculate and write velocity
	vec3 positionNDC      = fs_in.positionCS.xyz      / fs_in.positionCS.w;
	vec3 positionNDC_prev = fs_in.positionCS_prev.xyz / fs_in.positionCS_prev.w;
	// adjust for jitter!
	positionNDC.xy      -= uboMatUsr.mJitterCurrentPrev.xy;
	positionNDC_prev.xy -= uboMatUsr.mJitterCurrentPrev.zw;
	vec3 motionVector = (positionNDC - positionNDC_prev) * vec3(0.5, 0.5, 1.0); // TODO: check if z scale is ok
	oFragVelocity = vec4(motionVector, fs_in.movingObjectId);

	// Initialize all the colors:
	vec3 ambient    = materialsBuffer.materials[matIndex].mAmbientReflectivity.rgb  * diffTexColorRGBA.rgb;
	vec3 emissive   = materialsBuffer.materials[matIndex].mEmissiveColor.rgb        * emissiveTexColor; // TODO: check if we really want to multiply with emissive color (is typically vec3(0.5) for emerald square)
	vec3 diff       = materialsBuffer.materials[matIndex].mDiffuseReflectivity.rgb  * diffTexColorRGBA.rgb;
	vec3 spec       = materialsBuffer.materials[matIndex].mSpecularReflectivity.rgb * specTexValue;
	float shininess = materialsBuffer.materials[matIndex].mShininess;
	bool twoSided   = materialsBuffer.materials[matIndex].mCustomData[3] > 0.5;


	// Calculate ambient illumination:
	vec3 ambientIllumination = vec3(0.0, 0.0, 0.0);
	for (uint i = uboLights.mRangesAmbientDirectional[0]; i < uboLights.mRangesAmbientDirectional[1]; ++i) {
		ambientIllumination += uboLights.mLightData[i].mColor.rgb * ambient;
	}

	// Calculate diffuse and specular illumination from all light sources:
	vec3 diffAndSpecIllumination = calc_illumination_in_vs(positionVS, normalVS, diff, spec, shininess, twoSided);

	// Add all together:
	//vec4 blinnPhongColor = vec4(calc_shadow_factor(fs_in.positionWS) * vec3(ambientIllumination + emissive + diffAndSpecIllumination), alpha);
	vec4 blinnPhongColor = vec4(vec3(ambientIllumination + emissive + calc_shadow_factor(fs_in.positionWS) * diffAndSpecIllumination), alpha);

	if (uboMatUsr.mUserInput.z < 1.f) {
		oFragColor = blinnPhongColor;
	} else if (uboMatUsr.mUserInput.z < 2.f) {
		// don't use lights
		oFragColor = vec4(diff, alpha);
	} else if (uboMatUsr.mUserInput.z < 3.f) {
		// Debug: SM cascade
		oFragColor = debug_shadow_cascade_color() * blinnPhongColor;
	} else if (uboMatUsr.mUserInput.z < 4.f) {
		// Debug: show normals
		vec3 normalWS = normalize(mat3(transpose(uboMatUsr.mViewMatrix)) * normalVS);
		oFragColor = vec4(normalWS.xyz, 1.0);
	} else {
		// Debug2: show geometry normals (not affected by normal mapping)
		vec3 normalWS = normalize(mat3(inverse(transpose(fs_in.modelMatrix))) * normalize(fs_in.normalOS));
		oFragColor = vec4(normalWS.xyz, 1.0);

		//oFragColor = vec4(sample_from_normals_texture().rgb, 1.0); return;
	}
	//oFragColor = vec4(normalize(fs_in.normalOS.xyz) * 0.5 + 0.5, 1.0);
	//oFragColor = vec4(normalize(normalVS.xyz) * 0.5 + 0.5, 1.0);
}
// -------------------------------------------------------

