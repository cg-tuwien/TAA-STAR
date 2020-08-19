// ######################## uniforms ########################
uniform sampler2D uTexture;

layout (location = 0) in vec3 vSphereCoords;

// ################ output-color of fragment ################
layout (location = 0) out vec4 oFragColor;

void main()
{
	const float pi = 3.1415926535897932384626433832795;
	const float two_pi = pi * 2;
	vec3 nc = normalize(vSphereCoords);
	// assume uTexture to be a panoramic image in longitude latitude layout
	float latitude = acos(nc.y);
	float longitude = atan(nc.z, nc.x);
	vec2 sc = vec2(longitude / two_pi, latitude / pi);
	vec2 tc = vec2(0.5, 1.0) - sc;
	oFragColor = texture(uTexture, tc);
}
