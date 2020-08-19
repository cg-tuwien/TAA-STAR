#version 460

layout (location = 0) in vec4 aVertexPosition;

layout (location = 0) out vec3 vSphereCoords;

layout(push_constant) uniform PushConstants {
	mat4 pvmMatrix;
} pushConstants;

void main()
{
	vSphereCoords = aVertexPosition.xyz;
	vec4 position_cs = pushConstants.pvmMatrix * aVertexPosition;
	gl_Position = position_cs;
}
