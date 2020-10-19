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

// Unique push constants per draw call (You can think of
// these like single uniforms in OpenGL):
layout(push_constant) uniform PushConstants {
	mat4 mModelMatrix;
	int mMaterialIndex;
} pushConstants;

// "mMatrices" uniform buffer containing camera matrices:
// It is updated every frame.
layout(set = 1, binding = 0) UNIFORMDEF_MatricesAndUserInput uboMatUsr;

// -------------------------------------------------------

// ###### DATA PASSED ON ALONG THE PIPELINE ##############
// Data from vert -> tesc or frag:
layout (location = 0) out VertexData {
	vec3 positionOS;
	vec3 positionVS;
	vec2 texCoords;
	vec3 normalOS;
	vec3 tangentOS;
	vec3 bitangentOS;
	vec4 positionCS;		// TODO: don't really need this!
	vec4 positionCS_prev;	// position in previous frame
} v_out;
// -------------------------------------------------------

// ###### VERTEX SHADER MAIN #############################
void main()
{
	mat4 mMatrix = EFFECTIVE_MODELMATRIX;
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

	v_out.positionOS  = positionOS.xyz;
	v_out.positionVS  = positionVS.xyz;
	v_out.texCoords   = aTexCoords;
	v_out.normalOS    = normalOS;
	v_out.tangentOS   = tangentOS;
	v_out.bitangentOS = bitangentOS;
	v_out.positionCS  = positionCS;	// TODO: recheck - is it ok to interpolate clip space vars?
	v_out.positionCS_prev = uboMatUsr.mPrevFrameProjViewMatrix * EFFECTIVE_PREV_MODELMATRIX * positionOS;	// TODO: check if ok when mover gets enabled/disabled

	gl_Position = positionCS;
}
// -------------------------------------------------------

