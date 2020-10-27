#pragma once

#include <glm/glm.hpp>

// ---- Catmull Rom splines ( info source: https://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline )
template <typename T>
class CatmullRom {
public:
	float catmullRom_getTime(float t, T & p0, T & p1);
	T catmullRom_segment(T & p0, T & p1, T & p2, T & p3, float t);
	int catmullRom_getCurrentSegment(std::vector<T>& P, float t);
	T catmullRom_chain(std::vector<T>& P, float t, int *out_seg = nullptr, float *out_tseg = nullptr);
	float catmullRom_segmentLen(std::vector<T>& P, int idxPseg, int numSamples);
	float catmullRom_chainLen(std::vector<T>& P, int numSamplesPerSeg);
	std::vector<float> catmullRom_allSegmentLens(std::vector<T>& P, float & out_totalLen, int numSamplesPerSeg);

	float mAlpha = 0.5f; // 0.5 = centripetal catmull rom (0 = uniform, 1 = chordal)
};

struct Spline {
	float cam_t_max;
	std::vector<glm::vec3> camP;
	std::vector<glm::quat> camR;

	bool use_arclen;
	int rotation_mode = 1;	// 0 = ignore; 1 = lerp keyframe rot; 2 = from positions

	Spline(float cam_t_max, std::vector<glm::vec3> path) : cam_t_max(cam_t_max), camP(path) { camR.resize(camP.size(), glm::quat(1.f,0.f,0.f,0.f)); }
	void modified() { calced_arclen = false; }

	void interpolate(float t, glm::vec3 &pos, glm::quat &rot);
	float get_catmullrom_alpha() { return cmr_pos.mAlpha; }
	void  set_catmullrom_alpha(float alpha) { cmr_pos.mAlpha = alpha; }
private:
	float map_arclen_t(float spline_t);

	bool calced_arclen;
	std::vector<float> mSegLenNormalized;
	CatmullRom<glm::vec3> cmr_pos;
};

