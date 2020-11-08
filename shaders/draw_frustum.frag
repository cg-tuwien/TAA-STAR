#version 460

layout (location = 0) in  vec3 in_color;
layout (location = 0) out vec4 oFragColor;

void main() {
	oFragColor = vec4(in_color, 1);
}
