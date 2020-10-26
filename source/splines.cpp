#include "splines.hpp"

// ---- Catmull Rom splines ( info source: https://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline )
float catmullRom_getT(float t, glm::vec3 &p0, glm::vec3 &p1) {
	const float alpha = 0.5f; // 0.5 = centripetal catmull rom (0 = uniform, 1 = chordal)

							  // calc time for next control point
	return t + std::pow(glm::length(p1 - p0), alpha);
}
glm::vec3 catmullRom_segment(glm::vec3 &p0, glm::vec3 &p1, glm::vec3 &p2, glm::vec3 &p3, float t) {
	// t is the local time in the segment: 0 = p1, 1 = p2

	float t0 = 0.f;
	float t1 = catmullRom_getT(t0, p0, p1);
	float t2 = catmullRom_getT(t1, p1, p2);
	float t3 = catmullRom_getT(t2, p2, p3);

	t = t1 + t * (t2 - t1);

	glm::vec3 A1 = ((t1 - t) / (t1 - t0)) * p0 + ((t - t0) / (t1 - t0)) * p1;
	glm::vec3 A2 = ((t2 - t) / (t2 - t1)) * p1 + ((t - t1) / (t2 - t1)) * p2;
	glm::vec3 A3 = ((t3 - t) / (t3 - t2)) * p2 + ((t - t2) / (t3 - t2)) * p3;
	glm::vec3 B1 = ((t2 - t) / (t2 - t0)) * A1 + ((t - t0) / (t2 - t0)) * A2;
	glm::vec3 B2 = ((t3 - t) / (t3 - t1)) * A2 + ((t - t1) / (t3 - t1)) * A3;
	glm::vec3 C  = ((t2 - t) / (t2 - t1)) * B1 + ((t - t1) / (t2 - t1)) * B2;

	//printf("C = [%.2f %.2f %.2f]\n", C.x, C.y, C.z);
	return C;
}
int catmullRom_getCurrentSegment(std::vector<glm::vec3> &P, float t) {
	// which segment are we in?
	int numSeg = (int)P.size() - 3;
	assert(numSeg > 0);
	int curSeg = (int)floor(t * numSeg);
	if (curSeg >= numSeg) curSeg = numSeg-1;
	return curSeg;
}
glm::vec3 catmullRom_chain(std::vector<glm::vec3> &P, float t) {
	// t is the local time in the chain: 0 = P[1], 1 = P[<numP>-2]

	// which segment are we in?
	int curSeg = catmullRom_getCurrentSegment(P, t);

	int numSeg = (int)P.size() - 3;
	assert(numSeg > 0);

	float t_per_seg = 1.f / numSeg; // time per segment
	float t_seg =  (t - curSeg * t_per_seg) / t_per_seg; // local time in segment
	assert(t_seg >= 0.f && t_seg <= 1.f);

	return catmullRom_segment(P[curSeg], P[curSeg + 1], P[curSeg + 2], P[curSeg + 3], t_seg);
}
float catmullRom_segmentLen(std::vector<glm::vec3> &P, int idxPseg, int numSamples = 20) {
	float dt = 1.f / (float)numSamples;
	float sum = 0.f;
	for (int i = 0; i < numSamples; i++) {
		glm::vec3 a = catmullRom_segment(P[idxPseg - 1], P[idxPseg], P[idxPseg + 1], P[idxPseg + 2], dt * i);
		glm::vec3 b = catmullRom_segment(P[idxPseg - 1], P[idxPseg], P[idxPseg + 1], P[idxPseg + 2], dt * (i+1));
		sum += glm::length(b - a);
	}
	return sum;
}
float catmullRom_chainLen(std::vector<glm::vec3> &P, int numSamplesPerSeg = 20) {
	float sum = 0.f;
	for (int i = 0; i < P.size() - 3; i++) {
		sum += catmullRom_segmentLen(P, i + 1, numSamplesPerSeg);
	}
	return sum;
}
std::vector<float> catmullRom_allSegmentLens(std::vector<glm::vec3> &P, float &out_totalLen, int numSamplesPerSeg = 20) {
	std::vector<float> lens;
	out_totalLen = 0.f;
	for (int i = 0; i < P.size() - 3; i++) {
		float len = catmullRom_segmentLen(P, i + 1, numSamplesPerSeg);
		out_totalLen += len;
		lens.push_back(len);
	}
	return lens;
}

float Spline::map_arclen_t(float arc_t) {
	if (!calced_arclen) {
		int numSamples = 200;
		float totalLen;
		mSegLenNormalized = catmullRom_allSegmentLens(camP, totalLen, numSamples);
		if (totalLen > 0.f) for (auto &len : mSegLenNormalized) len /= totalLen;
		calced_arclen = true;
	}

	int numSeg = (int)camP.size() - 3;
	float t_per_seg = 1.f / numSeg; // time per segment

	float s = 0.f;
	for (int i = 0; i < int(mSegLenNormalized.size()); ++i) {
		float sNext = s + mSegLenNormalized[i];
		if (sNext > arc_t) {
			// found active segment
			float f = (sNext > s) ? (arc_t - s) / (sNext - s) : 0.5f;
			return (float(i) + f) * t_per_seg;
		}
		s = sNext;
	}

	return 1.f;
}

glm::vec3 Spline::getPos(float t) {
	t = std::max(0.f, std::min(1.f,  t / cam_t_max));
	if (use_arclen) {
		return catmullRom_chain(camP, map_arclen_t(t));
	} else {
		return catmullRom_chain(camP, t);
	}
}
