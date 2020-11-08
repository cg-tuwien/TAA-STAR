#version 460

layout(set = 0, binding = 5) uniform texture2D uShadowMap;
layout(set = 4, binding = 0) uniform sampler uSampler;

layout (location = 0) out vec4 oFragColor;

layout (location = 0) in VertexData {
	vec2 texCoords;
} f_in;

void main() {
	oFragColor = vec4(vec3(texture(sampler2D(uShadowMap, uSampler), f_in.texCoords).r), 1.0);
}
