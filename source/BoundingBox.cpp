#include "BoundingBox.hpp"

void BoundingBox::combineWith(const BoundingBox &other) {
	combineWith(other.min);
	combineWith(other.max);
}

void BoundingBox::combineWith(const glm::vec3 &point) {
	min = glm::min(min, point);
	max = glm::max(max, point);
}

float BoundingBox::getAbsMaxValue() {
	glm::vec3 v = glm::max(glm::abs(min), glm::abs(max));
	return glm::max(v.x, glm::max(v.y, v.z));
}

float BoundingBox::getLongestSide() {
	glm::vec3 v = max - min;
	return glm::max(v.x, glm::max(v.y, v.z));
}

void BoundingBox::calcFromPoints(size_t numPts, glm::vec3 * pPts) {
	if (numPts < 1) return;
	min = max = pPts[0];
	for (auto i = 1; i < numPts; i++) combineWith(pPts[i]);
}

void BoundingBox::calcFromPoints(size_t numPts, glm::vec4 * pPts) {
	if (numPts < 1) return;
	min = max = glm::vec3(pPts[0]);
	for (auto i = 1; i < numPts; i++) combineWith(glm::vec3(pPts[i]));
}

void BoundingBox::intersectWith(const BoundingBox &other) {
	min = glm::max(min, other.min);
	max = glm::min(max, other.max);
}

void BoundingBox::getPointsV4(glm::vec4 * p) {
	p[0] = glm::vec4(min.x, min.y, min.z, 1.0f);
	p[1] = glm::vec4(max.x, min.y, min.z, 1.0f);
	p[2] = glm::vec4(min.x, max.y, min.z, 1.0f);
	p[3] = glm::vec4(max.x, max.y, min.z, 1.0f);
	p[4] = glm::vec4(min.x, min.y, max.z, 1.0f);
	p[5] = glm::vec4(max.x, min.y, max.z, 1.0f);
	p[6] = glm::vec4(min.x, max.y, max.z, 1.0f);
	p[7] = glm::vec4(max.x, max.y, max.z, 1.0f);
}

void BoundingBox::getTransformedPointsV4(const glm::mat4 & transform, glm::vec4 * p) {
	p[0] = transform * glm::vec4(min.x, min.y, min.z, 1.0f);
	p[1] = transform * glm::vec4(max.x, min.y, min.z, 1.0f);
	p[2] = transform * glm::vec4(min.x, max.y, min.z, 1.0f);
	p[3] = transform * glm::vec4(max.x, max.y, min.z, 1.0f);
	p[4] = transform * glm::vec4(min.x, min.y, max.z, 1.0f);
	p[5] = transform * glm::vec4(max.x, min.y, max.z, 1.0f);
	p[6] = transform * glm::vec4(min.x, max.y, max.z, 1.0f);
	p[7] = transform * glm::vec4(max.x, max.y, max.z, 1.0f);
}
