#pragma once

#include <glm/glm.hpp>

struct BoundingBox {
	glm::vec3 min, max;

	void combineWith(const BoundingBox &other);
	void combineWith(const glm::vec3 &point);
	float getAbsMaxValue();
	float getLongestSide();
	void calcFromPoints(size_t numPts, glm::vec3 *pPts);
	void calcFromPoints(size_t numPts, glm::vec4 *pPts); // w-components are ignored
	void intersectWith(const BoundingBox &other);
	void getPointsV4(glm::vec4 *p);
	void getTransformedPointsV4(const glm::mat4 &transform, glm::vec4 *p);
};
