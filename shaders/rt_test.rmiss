#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 hitValue;

void main()
{
    //hitValue = vec3(1,0,1); return;

    // Skybox:
	const float pi = 3.1415926535897932384626433832795;
	const float half_pi = pi * 0.5;
	const float hdr_factor = 1.0;

	// Normalize direction vector towards the sky sphere:
	//vec3 nc = normalize(vSphereCoords);
	vec3 nc = gl_WorldRayDirectionEXT;

	// Compute the skybox color and store the result in oFragColor:
	float latitude = acos(nc.y);
	float longitude = atan(nc.z, nc.x);
	float s = cos(abs(longitude)) * 0.5 + 0.5;
	float t = mix(1.0 - nc.y, 0.5 + cos((nc.y + 0.5) * 0.5 * pi), 0.5);
	vec4 color = mix(vec4(0.1, 0.26, 0.4, 1.0), vec4(1.0, 0.68, 0.28, 1.0) * hdr_factor, t*t*t*t);
	float tf = clamp(latitude, 0.0, 1.0);
	color = mix(vec4(0.1, 0.26, 0.4, 1.0), color, s*s*s*s * tf);
	color = mix(color, vec4(0.0, 0.0, 0.5, 1.0), sin(clamp(-nc.y, 0.0, 1.0) * half_pi));

	hitValue = color.rgb;
}