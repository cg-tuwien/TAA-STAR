#version 460

// vSphereCoords represents the direction vector towards the sky sphere:
layout (location = 0) in vec3 vSphereCoords;

layout (location = 0) out vec4 oFragColor;

void main()
{
	const float pi = 3.1415926535897932384626433832795;
	const float half_pi = pi * 0.5;
	const float hdr_factor = 1.0;

	// Normalize direction vector towards the sky sphere:
	vec3 nc = normalize(vSphereCoords);

	// Compute the skybox color and store the result in oFragColor:
	float latitude = acos(nc.y);
	float longitude = atan(nc.z, nc.x);
	float s = cos(abs(longitude)) * 0.5 + 0.5;
	float t = mix(1.0 - nc.y, 0.5 + cos((nc.y + 0.5) * 0.5 * pi), 0.5);
	oFragColor = mix(vec4(0.1, 0.26, 0.4, 1.0), vec4(1.0, 0.68, 0.28, 1.0) * hdr_factor, t*t*t*t);
	float tf = clamp(latitude, 0.0, 1.0);
	oFragColor = mix(vec4(0.1, 0.26, 0.4, 1.0), oFragColor, s*s*s*s * tf);
	oFragColor = mix(oFragColor, vec4(0.0, 0.0, 0.5, 1.0), sin(clamp(-nc.y, 0.0, 1.0) * half_pi));
}

