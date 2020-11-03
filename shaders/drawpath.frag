#version 460

layout (location = 0) in flat float controlPointId;

layout (location = 0) out vec4 oFragColor;

void main() {
	if (controlPointId > 0) {
		oFragColor = vec4(0,1,0,1);
	} else {
		oFragColor = vec4(1,0,0,1);
	}
}
