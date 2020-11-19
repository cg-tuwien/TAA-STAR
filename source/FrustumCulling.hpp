#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>

// TODO: check https://gist.github.com/Kinwailo for a possibly faster method to extract the planes

class FrustumCulling {
private:
	glm::vec4	mPlanes[6];	// .xyz = normal, .w = distance

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

		mPlanes[0] = -(m3 + m0);
		mPlanes[1] = -(m3 - m0);
		mPlanes[2] = -(m3 + m1);
		mPlanes[3] = -(m3 - m1);
		mPlanes[4] = -(m3 + m2);
		mPlanes[5] = -(m3 - m2);

		if (normalize_planes) {
			for (int i = 0; i < 6; i++) {
				float len = length(vec3(mPlanes[i]));
				mPlanes[i] /= len;
			}
		}

	}

	// FrustumAABBIntersect code adapted from https://gist.github.com/Kinwailo
	// Returns: INTERSECT : 0 
	//          INSIDE : 1 
	//          OUTSIDE : 2 
	TestResult FrustumAABBIntersect(const glm::vec3 &mins, const glm::vec3 &maxs) const { 
		TestResult ret = TestResult::inside;
		glm::vec3  vmin, vmax; 

		for(int i = 0; i < 6; ++i) { 
			// X axis 
			if(mPlanes[i].x > 0) { 
				vmin.x = mins.x; 
				vmax.x = maxs.x; 
			} else { 
				vmin.x = maxs.x; 
				vmax.x = mins.x; 
			} 
			// Y axis 
			if(mPlanes[i].y > 0) { 
				vmin.y = mins.y; 
				vmax.y = maxs.y; 
			} else { 
				vmin.y = maxs.y; 
				vmax.y = mins.y; 
			} 
			// Z axis 
			if(mPlanes[i].z > 0) { 
				vmin.z = mins.z; 
				vmax.z = maxs.z; 
			} else { 
				vmin.z = maxs.z; 
				vmax.z = mins.z; 
			} 
			if(glm::dot(glm::vec3(mPlanes[i]), vmin) + mPlanes[i].w >  0.f) return TestResult::outside;
			if(glm::dot(glm::vec3(mPlanes[i]), vmax) + mPlanes[i].w >= 0.f) ret = TestResult::intersect;
		} 
		return ret;
	}

	bool CanCull(const BoundingBox &bb) const {
		return TestResult::outside == FrustumAABBIntersect(bb.min, bb.max);
	}

	glm::vec4 Plane(int index) { return mPlanes[index]; }
};
