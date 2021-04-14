#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : enable

#include "shader_cpu_common.h"
#include "shader_common_main.glsl"

layout(push_constant) PUSHCONSTANTSDEF_RAYTRACING pushConstants;

layout(location = 0) rayPayloadInEXT vec3 hitValue;
layout(location = 1) rayPayloadEXT float shadowHitValue;

layout(set = 0, binding = 0) BUFFERDEF_Material materialsBuffer;
layout(set = 0, binding = 1) uniform sampler2D textures[];
layout(set = 0, binding = 2) uniform usamplerBuffer indexBuffers[];     // one index buffer per [meshgroupId]; contents of each buffer are in uvec3s
layout(set = 0, binding = 3) uniform samplerBuffer  texCoordsBuffers[]; // ditto, entries are vec2  
layout (std430, set = 0, binding = 4) readonly buffer MaterialIndexBuffer { uint materialIndices[]; };
layout(set = 2, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 6) uniform samplerBuffer  normalsBuffers[];        // entries are vec3
//layout (std430, set = 0, binding = 8) readonly buffer AnimObjNormalsBuffer       { vec3 animObjNormals[]; }; // contains normals of all meshes of current anim object
layout (set = 0, binding = 8) uniform samplerBuffer animObjNormals; // contains normals of all meshes of current anim object
layout (std430, set = 0, binding = 9) readonly buffer AnimObjNormalsOffsetBuffer { uint animObjNrmOff[];  }; // .[meshIndex] = start index in animObjNormals[] for mesh meshIndex of current anim object

hitAttributeEXT vec2 bary2;

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


void main()
{
    // which index buffer to use? -> meshgroupId (stored in geometry custom index)
    int meshgroupId = gl_InstanceCustomIndexEXT;

    // calc texture coordinates by interpolating barycentric coordinates
    const vec3 barycentrics = vec3(1.0 - bary2.x - bary2.y, bary2);
    ivec3 indices = ivec3(texelFetch(indexBuffers[meshgroupId], gl_PrimitiveID).xyz);   // get the indices of the 3 triangle corners
    vec2 uv0 = texelFetch(texCoordsBuffers[meshgroupId], indices.x).xy;                 // and use them to look up the corresponding texture coordinates
    vec2 uv1 = texelFetch(texCoordsBuffers[meshgroupId], indices.y).xy;
    vec2 uv2 = texelFetch(texCoordsBuffers[meshgroupId], indices.z).xy;
    vec2 uv  = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;       // and interpolate

    uint matIndex = materialIndices[meshgroupId];

    // cast a shadow ray
    float shadowFactor;
    if ((pushConstants.mDoShadows & 0x01) != 0) {
        vec3 origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
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

    // get normal - this works different for animated objects
    bool isAnimObject = meshgroupId >= pushConstants.mAnimObjFirstMeshId && meshgroupId < (pushConstants.mAnimObjFirstMeshId + pushConstants.mAnimObjNumMeshes);
    vec3 n0,n1,n2;
    if (isAnimObject) {
        int meshIndex = meshgroupId - pushConstants.mAnimObjFirstMeshId;
        ivec3 newIndices = indices + int(animObjNrmOff[meshIndex]);
        n0 = texelFetch(animObjNormals, newIndices.x).xyz;
        n1 = texelFetch(animObjNormals, newIndices.y).xyz;
        n2 = texelFetch(animObjNormals, newIndices.z).xyz;
    } else {
        n0 = texelFetch(normalsBuffers[meshgroupId], indices.x).xyz;
        n1 = texelFetch(normalsBuffers[meshgroupId], indices.y).xyz;
        n2 = texelFetch(normalsBuffers[meshgroupId], indices.z).xyz;
    }
    vec3 normalOS = normalize(n0 * barycentrics.x + n1 * barycentrics.y + n2 * barycentrics.z);
    vec3 normalWS = normalize(gl_ObjectToWorldEXT * vec4(normalOS,0));   // note: gl_ObjectToWorldEXT is a mat4x3    // should we use inv transp ?

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
    //hitValue = normalOS * 0.5 + 0.5;

    // gl_PrimitiveID = triangle id. ; if multiple meshgroup-instances -> multiple triangles with same gl_PrimitiveID
    // gl_InstanceCustomIndexEXT = whatever is set in the GeometryInstance in user code

}