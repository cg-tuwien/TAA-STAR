#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : enable

#include "shader_common_main.glsl"

layout(push_constant) uniform PushConstants {
	mat4 mCameraTransform;
    vec4 mLightDir;
} pushConstants;

layout(location = 0) rayPayloadInEXT vec3 hitValue;
layout(location = 1) rayPayloadEXT float shadowHitValue;

layout(set = 0, binding = 0) BUFFERDEF_Material materialsBuffer;
layout(set = 0, binding = 1) uniform sampler2D textures[];
layout(set = 0, binding = 2) uniform usamplerBuffer indexBuffers[];     // one index buffer per [meshgroupId]; contents of each buffer are in uvec3s
layout(set = 0, binding = 3) uniform samplerBuffer  texCoordsBuffers[]; // ditto, entries are vec2  
layout (std430, set = 0, binding = 4) readonly buffer MaterialIndexBuffer { uint materialIndices[]; };
layout(set = 2, binding = 0) uniform accelerationStructureEXT topLevelAS;

const int NUMCOLOR = 2*6;
vec3 color[NUMCOLOR] = {
    {1.0,0,0}, {0,1.0,0}, {0,0,1.0}, {1.0,1.0,0}, {1.0,0,1.0}, {0,1.0,1.0},
    {0.3,0,0}, {0,0.3,0}, {0,0,0.3}, {0.3,0.3,0}, {0.3,0,0.3}, {0,0.3,0.3}
};

hitAttributeEXT vec2 bary2;

vec4 sample_from_diffuse_texture(uint matIndex, vec2 uv)
{
	int texIndex = materialsBuffer.materials[matIndex].mDiffuseTexIndex;
	vec4 offsetTiling = materialsBuffer.materials[matIndex].mDiffuseTexOffsetTiling;
	vec2 texCoords = uv * offsetTiling.zw + offsetTiling.xy;
	return texture(textures[texIndex], texCoords);
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
    vec2 uv = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;       // and interpolate

    uint materialIndex = materialIndices[meshgroupId];

    vec4 surfaceColor = sample_from_diffuse_texture(materialIndex, uv);

    // cast a shadow ray
    vec3 origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3 direction = normalize(-pushConstants.mLightDir.xyz); // mLightDir is the direction FROM the light source
    uint rayFlags = gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT;
    uint cullMask = 0xff;
    float tmin = 0.001;
    float tmax = 100.0;
    traceRayEXT(topLevelAS, rayFlags, cullMask, 1 /*sbtRecordOffset*/, 0 /*sbtRecordStride*/, 1 /*missIndex*/, origin, tmin, direction, tmax, 1 /*payload*/);

    hitValue = surfaceColor.rgb * (1.0 - 0.5 * shadowHitValue);

    //hitValue = vec3(1,0,0) * gl_HitTEXT / 100.0;
    //hitValue = color[gl_PrimitiveID % NUMCOLOR];
    //hitValue = vec3(clamp(gl_PrimitiveID/1000.0,0,1));
    //if (gl_PrimitiveID == 100) hitValue = vec3(1); else hitValue = vec3(0);
    //hitValue = color[gl_InstanceCustomIndexEXT % NUMCOLOR];
    //hitValue = barycentrics;
    //hitValue = vec3(uv,0);
    //hitValue = vec3(materialIndex[meshgroupId] / 100.0,0,0);


    // gl_PrimitiveID = triangle id. ; if multiple meshgroup-instances -> multiple triangles with same gl_PrimitiveID
    // gl_InstanceCustomIndexEXT = whatever is set in the GeometryInstance in user code

}