#version 460

layout(set = 0, binding = 5) uniform sampler2D shadowMap;
layout (location = 0) out vec4 oFragColor;

layout (location = 0) in VertexData {
	vec2 texCoords;
} f_in;

void main() {
	oFragColor = vec4(vec3(texture(shadowMap, f_in.texCoords).r), 1.0);
}
