#version 460
#extension GL_GOOGLE_include_directive : enable
// -------------------------------------------------------

#include "shader_common_main.glsl"

// ###### VERTEX SHADER/PIPELINE INPUT DATA ##############
// Several vertex attributes (These are the buffers passed
// to command_buffer_t::draw_indexed in the same order):
layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec2 aTexCoords;
layout (location = 2) in vec3 aNormal;
layout (location = 3) in vec3 aTangent;
layout (location = 4) in vec3 aBitangent;

struct PerInstanceAttribute { mat4 modelMatrix; };
layout (std430, set = 0, binding = 2) readonly buffer MaterialIndexBuffer   { uint materialIndex[]; };			// per meshgroup
layout (std430, set = 0, binding = 3) readonly buffer AttribBaseIndexBuffer { uint attrib_base[]; };			// per meshgroup
layout (std430, set = 0, binding = 4) readonly buffer mAttributesBuffer     { PerInstanceAttribute attrib[]; };	// per mesh

layout(push_constant) uniform PushConstantsDII { int mDrawIdOffset; };

// "mMatrices" uniform buffer containing camera matrices:
// It is updated every frame.
layout(set = 1, binding = 0) UNIFORMDEF_MatricesAndUserInput uboMatUsr;

// -------------------------------------------------------

// ###### DATA PASSED ON ALONG THE PIPELINE ##############
// Data from vert -> tesc or frag:
layout (location = 0) out VertexData {
	flat int drawID;		// gl_DrawID (already corrected by mDrawIdOffset) and
	flat int instanceIndex;	// gl_InstanceIndex only exist in the vertex shader
	vec3 positionOS;
	vec3 positionVS;
	vec2 texCoords;
	vec3 normalOS;
	vec3 tangentOS;
	vec3 bitangentOS;
} v_out;
// -------------------------------------------------------

// ###### VERTEX SHADER MAIN #############################
void main()
{
	uint attribBase = attrib_base[gl_DrawID + mDrawIdOffset];
	uint attribIndex = attribBase + gl_InstanceIndex;

	mat4 mMatrix = attrib[attribIndex].modelMatrix;
	mat4 vMatrix = uboMatUsr.mViewMatrix;
	mat4 pMatrix = uboMatUsr.mProjMatrix;
	mat4 vmMatrix = vMatrix * mMatrix;
	mat4 pvmMatrix = pMatrix * vmMatrix;

	vec4 positionOS  = vec4(aPosition, 1.0);
	vec4 positionVS  = vmMatrix * positionOS;
	vec4 positionCS  = pMatrix * positionVS;
	vec3 normalOS    = normalize(aNormal);
	vec3 tangentOS   = normalize(aTangent);
	vec3 bitangentOS = normalize(aBitangent);

	v_out.drawID      = gl_DrawID + mDrawIdOffset;
	v_out.positionOS  = positionOS.xyz;
	v_out.positionVS  = positionVS.xyz;
	v_out.texCoords   = aTexCoords;
	v_out.normalOS    = normalOS;
	v_out.tangentOS   = tangentOS;
	v_out.bitangentOS = bitangentOS;

	gl_Position = positionCS;
}
// -------------------------------------------------------

