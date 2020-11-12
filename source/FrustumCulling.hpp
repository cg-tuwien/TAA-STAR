#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>

// TODO: check https://gist.github.com/Kinwailo for a possibly faster method to extract the planes

class FrustumCulling {
private:
	glm::vec3	mPlaneN[6];
	float		mPlaneD[6];

	const static bool normalize_planes = false; // not needed
public:
	enum class TestResult { inside, outside, intersect };

	FrustumCulling(const glm::mat4 &projViewMatrix) {
		using namespace glm;
		// see Real-Time Rendering 4th ed., pp. 984 for derivation of frustrum planes from the projViewMatrix m
		// left   -(m3, + m0,)
		// right  -(m3, - m0,)
		// bottom -(m3, + m1,)
		// top    -(m3, - m1,)
		// near   -(m3, + m2,)
		// far    -(m3, - m2,)

		// calculate normal vectors and offset for frustum planes (note mi, denotes i'th ROW above - that is m[.][i] in a glm::mat4)
		vec4 m0 = glm::row(projViewMatrix, 0);
		vec4 m1 = glm::row(projViewMatrix, 1);
		vec4 m2 = glm::row(projViewMatrix, 2);
		vec4 m3 = glm::row(projViewMatrix, 3);

		vec4 n4[6];
		n4[0] = -(m3 + m0);
		n4[1] = -(m3 - m0);
		n4[2] = -(m3 + m1);
		n4[3] = -(m3 - m1);
		n4[4] = -(m3 + m2);
		n4[5] = -(m3 - m2);

		if (normalize_planes) {
			for (int i = 0; i < 6; i++) {
				float len = length(vec3(n4[i]));
				n4[i] /= len;
			}
		}

		for (int i = 0; i < 6; i++) {
			mPlaneN[i] = vec3(n4[i]);
			mPlaneD[i] = n4[i].w;
		}
	}

	// FrustumAABBIntersect code adapted from https://gist.github.com/Kinwailo
	// Returns: INTERSECT : 0 
	//          INSIDE : 1 
	//          OUTSIDE : 2 
	TestResult FrustumAABBIntersect(glm::vec3 &mins, glm::vec3 &maxs) { 
		TestResult ret = TestResult::inside;
		glm::vec3  vmin, vmax; 

		for(int i = 0; i < 6; ++i) { 
			// X axis 
			if(mPlaneN[i].x > 0) { 
				vmin.x = mins.x; 
				vmax.x = maxs.x; 
			} else { 
				vmin.x = maxs.x; 
				vmax.x = mins.x; 
			} 
			// Y axis 
			if(mPlaneN[i].y > 0) { 
				vmin.y = mins.y; 
				vmax.y = maxs.y; 
			} else { 
				vmin.y = maxs.y; 
				vmax.y = mins.y; 
			} 
			// Z axis 
			if(mPlaneN[i].z > 0) { 
				vmin.z = mins.z; 
				vmax.z = maxs.z; 
			} else { 
				vmin.z = maxs.z; 
				vmax.z = mins.z; 
			} 
			if(glm::dot(mPlaneN[i], vmin) + mPlaneD[i] >  0.f) return TestResult::outside;
			if(glm::dot(mPlaneN[i], vmax) + mPlaneD[i] >= 0.f) ret = TestResult::intersect;
		} 
		return ret;
	} 
};
