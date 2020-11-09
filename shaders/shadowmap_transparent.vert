#version 460
#extension GL_GOOGLE_include_directive : enable
// -------------------------------------------------------

#include "shader_common_main.glsl"
#include "shader_cpu_common.h"

// Note: this is the only shader in shadowmap generation for non-transparent objects (no frag shader)

// ###### VERTEX SHADER/PIPELINE INPUT DATA ##############
layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec2 aTexCoords;

struct PerInstanceAttribute { mat4 modelMatrix; };
layout (std430, set = 0, binding = 2) readonly buffer MaterialIndexBuffer   { uint materialIndex[]; };			// per meshgroup
layout (std430, set = 0, binding = 3) readonly buffer AttribBaseIndexBuffer { uint attrib_base[]; };			// per meshgroup
layout (std430, set = 0, binding = 4) readonly buffer mAttributesBuffer     { PerInstanceAttribute attrib[]; };	// per mesh

// push constants
layout(push_constant) PUSHCONSTANTSDEF_DII;

// matrices
layout(set = 1, binding = 0) UNIFORMDEF_MatricesAndUserInput uboMatUsr;

// ###### DATA PASSED ON ALONG THE PIPELINE ##############
layout (location = 0) out VertexData {
	vec2 texCoords;
	flat uint materialIndex;
} v_out;

// -------------------------------------------------------

// ###### VERTEX SHADER MAIN #############################
void main()
{
	mat4 modelMatrix;
	if (mDrawIdOffset >= 0) {
		// static scenery
		uint meshgroup = gl_DrawID + mDrawIdOffset;
		uint attribIndex = attrib_base[meshgroup] + gl_InstanceIndex;
		v_out.materialIndex  = materialIndex[meshgroup];
		modelMatrix    = attrib[attribIndex].modelMatrix;
	} else {
		// moving object
		v_out.materialIndex = mMover_materialIndex;
		modelMatrix   = uboMatUsr.mMover_additionalModelMatrix * mMover_baseModelMatrix;
	}

	gl_Position = uboMatUsr.mShadowmapProjViewMatrix[mShadowMapCascadeToBuild] * modelMatrix * vec4(aPosition, 1.0);
	v_out.texCoords   = aTexCoords;
}
// -------------------------------------------------------

