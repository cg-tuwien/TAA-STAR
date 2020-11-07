#version 460
#extension GL_GOOGLE_include_directive : enable
// -------------------------------------------------------

#include "shader_common_main.glsl"
#include "shader_cpu_common.h"

// ###### VERTEX SHADER/PIPELINE INPUT DATA ##############
layout (location = 0) in vec3 aPosition;

struct PerInstanceAttribute { mat4 modelMatrix; };
layout (std430, set = 0, binding = 2) readonly buffer MaterialIndexBuffer   { uint materialIndex[]; };			// per meshgroup
layout (std430, set = 0, binding = 3) readonly buffer AttribBaseIndexBuffer { uint attrib_base[]; };			// per meshgroup
layout (std430, set = 0, binding = 4) readonly buffer mAttributesBuffer     { PerInstanceAttribute attrib[]; };	// per mesh

layout(push_constant) uniform PushConstantsDII {
	mat4  mMover_baseModelMatrix;
	int   mMover_materialIndex;
	int   mMover_meshIndex;

	int   mDrawIdOffset; // negative numbers -> moving object id
	float pad1;
};


// "mMatrices" uniform buffer containing camera matrices:
// It is updated every frame.
layout(set = 1, binding = 0) uniform ShadowmapMatrices {
	mat4 mProjViewMatrix;
	mat4 mMover_additionalModelMatrix;
} uboMatUsr;

// -------------------------------------------------------

// ###### VERTEX SHADER MAIN #############################
void main()
{
	mat4 modelMatrix;
	if (mDrawIdOffset >= 0) {
		// static scenery
		uint meshgroup = gl_DrawID + mDrawIdOffset;
		uint attribIndex = attrib_base[meshgroup] + gl_InstanceIndex;
		modelMatrix    = attrib[attribIndex].modelMatrix;
	} else {
		// moving object
		modelMatrix   = uboMatUsr.mMover_additionalModelMatrix * mMover_baseModelMatrix;
	}

	gl_Position = uboMatUsr.mProjViewMatrix * modelMatrix * vec4(aPosition, 1.0);
}
// -------------------------------------------------------

