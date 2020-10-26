#include "splines.hpp"


// ---- Catmull Rom splines ( info source: https://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline )
template <typename T>
float CatmullRom<T>::catmullRom_getTime(float t, T &p0, T &p1) {
	const float alpha = 0.5f; // 0.5 = centripetal catmull rom (0 = uniform, 1 = chordal)

	// calc time for next control point
	return t + std::pow(glm::length(p1 - p0), alpha);
}

template <typename T>
T CatmullRom<T>::catmullRom_segment(T &p0, T &p1, T &p2, T &p3, float t) {
	// t is the local time in the segment: 0 = p1, 1 = p2

	float t0 = 0.f;
	float t1 = catmullRom_getTime(t0, p0, p1);
	float t2 = catmullRom_getTime(t1, p1, p2);
	float t3 = catmullRom_getTime(t2, p2, p3);

	t = t1 + t * (t2 - t1);

	//T A1 = ((t1 - t) / (t1 - t0)) * p0 + ((t - t0) / (t1 - t0)) * p1;
	//T A2 = ((t2 - t) / (t2 - t1)) * p1 + ((t - t1) / (t2 - t1)) * p2;
	//T A3 = ((t3 - t) / (t3 - t2)) * p2 + ((t - t2) / (t3 - t2)) * p3;
	//T B1 = ((t2 - t) / (t2 - t0)) * A1 + ((t - t0) / (t2 - t0)) * A2;
	//T B2 = ((t3 - t) / (t3 - t1)) * A2 + ((t - t1) / (t3 - t1)) * A3;
	//T C  = ((t2 - t) / (t2 - t1)) * B1 + ((t - t1) / (t2 - t1)) * B2;

	// FIXME!!
	float f0 = t1 - t0;
	float f1 = t2 - t1;
	float f2 = t3 - t2;
	float f3 = t2 - t0;
	float f4 = t3 - t1;
	float f5 = t2 - t1;
	const float eps = (float)1e-6;
	if (f0 < eps || f1 < eps || f2 < eps || f3 < eps || f4 < eps || f5 < eps) return p1;
	T A1 = ((t1 - t) / f0) * p0 + ((t - t0) / f0) * p1;
	T A2 = ((t2 - t) / f1) * p1 + ((t - t1) / f1) * p2;
	T A3 = ((t3 - t) / f2) * p2 + ((t - t2) / f2) * p3;
	T B1 = ((t2 - t) / f3) * A1 + ((t - t0) / f3) * A2;
	T B2 = ((t3 - t) / f4) * A2 + ((t - t1) / f4) * A3;
	T C  = ((t2 - t) / f5) * B1 + ((t - t1) / f5) * B2;


	//printf("C = [%.2f %.2f %.2f]\n", C.x, C.y, C.z);
	return C;
}

template <typename T>
int CatmullRom<T>::catmullRom_getCurrentSegment(std::vector<T> &P, float t) {
	// which segment are we in?
	int numSeg = (int)P.size() - 3;
	assert(numSeg > 0);
	int curSeg = (int)floor(t * numSeg);
	if (curSeg >= numSeg) curSeg = numSeg-1;
	return curSeg;
}

template <typename T>
T CatmullRom<T>::catmullRom_chain(std::vector<T> &P, float t, int *out_seg, float *out_tseg) {
	// t is the local time in the chain: 0 = P[1], 1 = P[<numP>-2]

	// which segment are we in?
	int curSeg = catmullRom_getCurrentSegment(P, t);

	int numSeg = (int)P.size() - 3;
	assert(numSeg > 0);

	float t_per_seg = 1.f / numSeg; // time per segment
	float t_seg =  (t - curSeg * t_per_seg) / t_per_seg; // local time in segment
	assert(t_seg >= 0.f && t_seg <= 1.f);

	if (out_seg)  *out_seg  = curSeg + 1;
	if (out_tseg) *out_tseg = t_seg;

	return catmullRom_segment(P[curSeg], P[curSeg + 1], P[curSeg + 2], P[curSeg + 3], t_seg);
}

template <typename T>
float CatmullRom<T>::catmullRom_segmentLen(std::vector<T> &P, int idxPseg, int numSamples) {
	float dt = 1.f / (float)numSamples;
	float sum = 0.f;
	for (int i = 0; i < numSamples; i++) {
		T a = catmullRom_segment(P[idxPseg - 1], P[idxPseg], P[idxPseg + 1], P[idxPseg + 2], dt * i);
		T b = catmullRom_segment(P[idxPseg - 1], P[idxPseg], P[idxPseg + 1], P[idxPseg + 2], dt * (i+1));
		sum += glm::length(b - a);
	}
	return sum;
}

template <typename T>
float CatmullRom<T>::catmullRom_chainLen(std::vector<T> &P, int numSamplesPerSeg) {
	float sum = 0.f;
	for (int i = 0; i < P.size() - 3; i++) {
		sum += catmullRom_segmentLen(P, i + 1, numSamplesPerSeg);
	}
	return sum;
}

template <typename T>
std::vector<float> CatmullRom<T>::catmullRom_allSegmentLens(std::vector<T> &P, float &out_totalLen, int numSamplesPerSeg) {
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
		mSegLenNormalized = cmr_pos.catmullRom_allSegmentLens(camP, totalLen, numSamples);
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

void Spline::interpolate(float t, glm::vec3 &pos, glm::quat &rot) {
	t = std::max(0.f, std::min(1.f, t ));
	if (use_arclen) t = map_arclen_t(t);
	int seg;
	float tseg;
	pos = cmr_pos.catmullRom_chain(camP, t, &seg, &tseg);

	// just lerp rot
	rot = glm::mix(camR[seg], camR[seg + 1], tseg);
}

