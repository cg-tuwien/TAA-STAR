#version 460

layout (location = 0) in vec3 aVertexPosition;

layout (location = 0) out vec3 vSphereCoords;

layout(set = 0, binding = 0) uniform MatricesAndUserInput {
	// view matrix as returned from quake_camera
	mat4 mViewMatrix;
	// projection matrix as returned from quake_camera
	mat4 mProjMatrix;
	// transformation matrix which tranforms to camera's position
	mat4 mCamPos;
	// x = tessellation factor, y = displacement strength, z and w unused
	vec4 mUserInput;
} uboMatUsr;


void main()
{
	vSphereCoords = aVertexPosition.xyz;
	vec4 position_cs = uboMatUsr.mProjMatrix * uboMatUsr.mViewMatrix * uboMatUsr.mCamPos * vec4(aVertexPosition, 1.0);
	gl_Position = position_cs;
}

