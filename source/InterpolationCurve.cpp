#include "InterpolationCurve.hpp"

void InterpolationCurve::setType(InterpolationCurveType aType) {
	if (aType == mType) return;

	// save control points
	std::vector<glm::vec3> oldPts = interpolator().control_points();
	mType = aType;
	interpolator().set_control_points(oldPts);

	invalidateArcLenTable();
}

gvk::cp_interpolation & InterpolationCurve::interpolator()
{
	switch (mType) {
	case InterpolationCurveType::bezier_curve:					return mInterpolatorBezier;
	case InterpolationCurveType::quadratic_uniform_b_spline:	return mInterpolatorQuadB;
	case InterpolationCurveType::cubic_uniform_b_spline:		return mInterpolatorCubeB;
	case InterpolationCurveType::catmull_rom_spline:			return mInterpolatorCatmull;
	}
	throw avk::runtime_error("Invalid interpolation curve type");
}

void InterpolationCurve::set_control_points(std::vector<glm::vec3> pControlPoints) { interpolator().set_control_points(pControlPoints); invalidateArcLenTable(); }
const std::vector<glm::vec3>& InterpolationCurve::control_points() { return interpolator().control_points(); }
const glm::vec3& InterpolationCurve::control_point_at(size_t index) { return interpolator().control_point_at(index); }
size_t InterpolationCurve::num_control_points() { return interpolator().control_points().size(); }
glm::vec3 InterpolationCurve::value_at(float t) { return valid() ? interpolator().value_at(t) : glm::vec3(0); }
glm::vec3 InterpolationCurve::slope_at(float t) { return valid() ? interpolator().slope_at(t) : glm::vec3(0); }

bool InterpolationCurve::valid()
{
	if (mType == InterpolationCurveType::catmull_rom_spline) {
		return interpolator().num_control_points() >= 4;
	} else {
		return interpolator().num_control_points() >= 2;
	}
}

void InterpolationCurve::buildArcLenTable()
{
	int nSegs = static_cast<int>((mType == InterpolationCurveType::catmull_rom_spline) ? num_control_points() - 3 : num_control_points() - 1);
	int nSamples = mArcLenSamplesPerSegment * nSegs;

	if (!valid() || nSamples < 2) return;

	mArcLenTable.resize(nSamples);
	float distance = 0.f;
	auto prev = value_at(0.f);
	mArcLenTable[0] = distance;
	for (int i = 1; i < nSamples; ++i) {
		float t = static_cast<float>(i) / static_cast<float>(nSamples - 1);
		auto next = value_at(t);
		distance += glm::length(next - prev);
		mArcLenTable[i] = distance;
		prev = next;
	}
	if (distance > 0.f) {
		for (int i = 0; i < nSamples; ++i) mArcLenTable[i] /= distance;
	}
}

void InterpolationCurve::invalidateArcLenTable()
{
	mArcLenTable.clear();
}


float InterpolationCurve::mapConstantSpeedTime(float t)
{
	if (mArcLenTable.empty()) buildArcLenTable();
	if (mArcLenTable.empty()) return t;

	int nSamples = static_cast<int>(mArcLenTable.size());

	if (t < mArcLenTable[0]) return 0.f;
	if (t > mArcLenTable[nSamples-1]) return 1.f;

	// binary search in normalized arc len table; find first entry with value > t
	int lo = 0;
	int hi = nSamples - 1;
	while (lo + 1 < hi) {
		int mid = (lo + hi) / 2;
		if (mArcLenTable[mid] < t) {
			lo = mid;
		} else {
			hi = mid;
		}
	}
	assert(lo + 1 == hi);
	assert(mArcLenTable[lo] <= t);
	assert(mArcLenTable[hi] >= t);

	float f = (mArcLenTable[lo] == mArcLenTable[hi]) ? 0.f : (t - mArcLenTable[lo]) / (mArcLenTable[hi] - mArcLenTable[lo]);
	float t1 = static_cast<float>(lo) / static_cast<float>(nSamples - 1);
	float t2 = static_cast<float>(hi) / static_cast<float>(nSamples - 1);
	return t1 * (1.f - f) + t2 * f;
}

