#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : enable

#include "shader_cpu_common.h"
#include "shader_common_main.glsl"

layout(push_constant) PUSHCONSTANTSDEF_RAYTRACING pushConstants;

layout(location = 0) rayPayloadInEXT vec3 hitValue;
layout(location = 1) rayPayloadEXT float shadowHitValue;

layout(set = 0, binding =  0) BUFFERDEF_Material materialsBuffer;
layout(set = 0, binding =  1) uniform sampler2D textures[];
layout(set = 0, binding =  2) uniform usamplerBuffer indexBuffers[];     // one index buffer per [meshgroupId]; contents of each buffer are in uvec3s
layout(set = 0, binding =  3) uniform samplerBuffer  texCoordsBuffers[]; // ditto, entries are vec2  
layout (std430, set = 0, binding = 4) readonly buffer MaterialIndexBuffer { uint materialIndices[]; };
layout(set = 2, binding =  0) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding =  6) uniform samplerBuffer  normalsBuffers[];        // entries are vec3
layout(set = 0, binding = 10) uniform samplerBuffer  tangentsBuffers[];       // entries are vec3
layout(set = 0, binding = 11) uniform samplerBuffer  bitangentsBuffers[];     // entries are vec3
//layout (std430, set = 0, binding = 8) readonly buffer AnimObjNormalsBuffer       { vec3 animObjNormals[]; }; // contains normals of all meshes of current anim object
layout(set = 0, binding =  8) uniform samplerBuffer animObjNormals;  // contains normals of all meshes of current anim object
layout(set = 0, binding = 12) uniform samplerBuffer animObjTangents;
layout(set = 0, binding = 13) uniform samplerBuffer animObjBitangents;
layout(std430, set = 0, binding = 9) readonly buffer AnimObjNTBOffsetBuffer { uint animObjNTBOff[];  }; // .[meshIndex] = start index in animObjNormals[] for mesh meshIndex of current anim object

hitAttributeEXT vec2 bary2;

mat3 matrixNormalsOStoWS;	// matrix to bring normals from object to world space

vec4 sample_from_normals_texture(uint matIndex, vec2 uv)
{
	int texIndex = materialsBuffer.materials[matIndex].mNormalsTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mNormalsTexOffsetTiling;
	vec2 texCoords = uv * offsetTiling.zw + offsetTiling.xy;
	vec4 normalSample = texture(textures[texIndex], texCoords);
	FIX_NORMALMAPPING(normalSample);
	return normalSample;
}

vec4 sample_from_diffuse_texture(uint matIndex, vec2 uv)
{
	int texIndex = materialsBuffer.materials[matIndex].mDiffuseTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mDiffuseTexOffsetTiling;
	vec2 texCoords = uv * offsetTiling.zw + offsetTiling.xy;
	return texture(textures[texIndex], texCoords);
}

vec4 sample_from_specular_texture(uint matIndex, vec2 uv)
{
	int texIndex = materialsBuffer.materials[matIndex].mSpecularTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mSpecularTexOffsetTiling;
	vec2 texCoords = uv * offsetTiling.zw + offsetTiling.xy;
	return texture(textures[texIndex], texCoords);
}

vec4 sample_from_emissive_texture(uint matIndex, vec2 uv)
{
	int texIndex = materialsBuffer.materials[matIndex].mEmissiveTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mEmissiveTexOffsetTiling;
	vec2 texCoords = uv * offsetTiling.zw + offsetTiling.xy;
	return texture(textures[texIndex], texCoords);
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

// Re-orthogonalizes the first vector w.r.t. the second vector (Gram-Schmidt process)
vec3 re_orthogonalize(vec3 first, vec3 second)
{
	return normalize(first - dot(first, second) * second);
}

// Calculates the normalized normal in world space by sampling the
// normal from the normal map and transforming it with the TBN-matrix.
// input vectors need to be normalized already
vec3 calc_normalized_normalWS(vec3 sampledNormal, in vec3 normalOS, in vec3 tangentOS, in vec3 bitangentOS, uint matIndex)
{
//	mat4 vmMatrix = uboMatUsr.mViewMatrix * fs_in.modelMatrix;
//	mat3 vmNormalMatrix = mat3(inverse(transpose(vmMatrix)));

	// build the TBN matrix from the varyings
	tangentOS   = re_orthogonalize(tangentOS,   normalOS);
	bitangentOS = re_orthogonalize(bitangentOS, normalOS);

	mat3 matrixTStoOS = inverse(transpose(mat3(tangentOS, bitangentOS, normalOS)));

	// sample the normal from the normal map and bring it into view space
	vec3 normalSample = normalize(sampledNormal * 2.0 - 1.0);

	float normalMappingStrengthFactor = 1.0f - materialsBuffer.materials[matIndex].mCustomData[2];

	float userDefinedDisplacementStrength = pushConstants.mNormalMappingStrength;
	normalSample.xy *= userDefinedDisplacementStrength * normalMappingStrengthFactor;

	vec3 normalWS = matrixNormalsOStoWS * matrixTStoOS * normalSample;

	return normalize(normalWS);
}


#define INTERPOL_BARY(a_,b_,c_) ((a_) * barycentrics.x + (b_) * barycentrics.y + (c_) * barycentrics.z)
#define INTERPOL_BARY_TEXELFETCH(tex_,indices_,swz_) INTERPOL_BARY(texelFetch((tex_), (indices_).x).swz_, texelFetch((tex_), (indices_).y).swz_, texelFetch((tex_), (indices_).z).swz_)

void main()
{
	matrixNormalsOStoWS = inverse(transpose(mat3(gl_ObjectToWorldEXT)));

    // which index buffer to use? -> meshgroupId (stored in geometry custom index)
    int meshgroupId = gl_InstanceCustomIndexEXT;

    // calc texture coordinates by interpolating barycentric coordinates
    const vec3 barycentrics = vec3(1.0 - bary2.x - bary2.y, bary2);
    ivec3 indices = ivec3(texelFetch(indexBuffers[meshgroupId], gl_PrimitiveID).xyz);   // get the indices of the 3 triangle corners
    vec2 uv = INTERPOL_BARY_TEXELFETCH(texCoordsBuffers[meshgroupId], indices, xy);     // and interpolate

    uint matIndex = materialIndices[meshgroupId];

    // get normal, tangent, bitangent - this works different for animated objects
    bool isAnimObject = meshgroupId >= pushConstants.mAnimObjFirstMeshId && meshgroupId < (pushConstants.mAnimObjFirstMeshId + pushConstants.mAnimObjNumMeshes);
    vec3 normalWS, normalOS, tangentOS, bitangentOS;
    if (isAnimObject) {
        int meshIndex = meshgroupId - pushConstants.mAnimObjFirstMeshId;
        ivec3 newIndices = indices + int(animObjNTBOff[meshIndex]);

        normalOS    = normalize(INTERPOL_BARY_TEXELFETCH(animObjNormals,    newIndices, xyz));
        tangentOS   = normalize(INTERPOL_BARY_TEXELFETCH(animObjTangents,   newIndices, xyz));
        bitangentOS = normalize(INTERPOL_BARY_TEXELFETCH(animObjBitangents, newIndices, xyz));
    } else {
        normalOS    = normalize(INTERPOL_BARY_TEXELFETCH(normalsBuffers   [meshgroupId], indices, xyz));
        tangentOS   = normalize(INTERPOL_BARY_TEXELFETCH(tangentsBuffers  [meshgroupId], indices, xyz));
        bitangentOS = normalize(INTERPOL_BARY_TEXELFETCH(bitangentsBuffers[meshgroupId], indices, xyz));
    }
    //normalWS = normalize(matrixNormalsOStoWS * normalOS);
    normalWS = calc_normalized_normalWS(sample_from_normals_texture(matIndex, uv).rgb, normalOS, tangentOS, bitangentOS, matIndex);

    // cast a shadow ray
    float shadowFactor;
    if ((pushConstants.mDoShadows & 0x01) != 0) {
        vec3 origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
        //origin += normalWS * 0.1; // just testing - this helps a bit against the self-shadows on goblin; however 0.1 seems a bit much for a general offset
        vec3 direction = normalize(-pushConstants.mLightDir.xyz); // mLightDir is the direction FROM the light source
        //uint rayFlags = gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT;
        uint rayFlags = gl_RayFlagsCullBackFacingTrianglesEXT | gl_RayFlagsTerminateOnFirstHitEXT;
        uint cullMask = (pushConstants.mDoShadows & 0x02) != 0 ? RAYTRACING_CULLMASK_OPAQUE | RAYTRACING_CULLMASK_TRANSPARENT : RAYTRACING_CULLMASK_OPAQUE;
        float tmin = 0.001;
        float tmax = pushConstants.mMaxRayLength; //100.0;
        traceRayEXT(topLevelAS, rayFlags, cullMask, 1 /*sbtRecordOffset*/, 0 /*sbtRecordStride*/, 1 /*missIndex*/, origin, tmin, direction, tmax, 1 /*payload*/);
        shadowFactor = 1.0 - SHADOW_OPACITY * shadowHitValue;
    } else {
        shadowFactor = 1.0;
    }

    // simple light calculation
    // we only support one directional light here (+ ambient)

    vec4 diffTexColorRGBA  = sample_from_diffuse_texture (matIndex, uv);
	float specTexValue     = sample_from_specular_texture(matIndex, uv).r;
	vec3  emissiveTexColor = sample_from_emissive_texture(matIndex, uv).rgb;

	vec3 ambient    = materialsBuffer.materials[matIndex].mAmbientReflectivity.rgb  * diffTexColorRGBA.rgb;
	vec3 emissive   = materialsBuffer.materials[matIndex].mEmissiveColor.rgb        * emissiveTexColor; // TODO: check if we really want to multiply with emissive color (is typically vec3(0.5) for emerald square)
	vec3 diff       = materialsBuffer.materials[matIndex].mDiffuseReflectivity.rgb  * diffTexColorRGBA.rgb;
	vec3 spec       = materialsBuffer.materials[matIndex].mSpecularReflectivity.rgb * specTexValue;
	float shininess = materialsBuffer.materials[matIndex].mShininess;
	bool twoSided   = materialsBuffer.materials[matIndex].mCustomData[3] > 0.5;

	vec3 ambientIllumination = pushConstants.mAmbientLightIntensity.rgb * ambient;
    vec3 dirLightIntensity   = pushConstants.mDirLightIntensity.rgb;
    vec3 toLightWS = normalize(-pushConstants.mLightDir.xyz);
    vec3 toEyeWS   = -gl_WorldRayDirectionEXT;
    vec3 diffAndSpec = dirLightIntensity * calc_blinn_phong_contribution(toLightWS, toEyeWS, normalWS, diff, spec, shininess, twoSided);
	vec4 blinnPhongColor = vec4(vec3(ambientIllumination + emissive + shadowFactor * diffAndSpec), 1.0);

    hitValue = blinnPhongColor.rgb;

    // gl_PrimitiveID = triangle id. ; if multiple meshgroup-instances -> multiple triangles with same gl_PrimitiveID
    // gl_InstanceCustomIndexEXT = whatever is set in the GeometryInstance in user code

}