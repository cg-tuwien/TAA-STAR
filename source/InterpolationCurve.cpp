#include "InterpolationCurve.hpp"

void InterpolationCurve::setType(InterpolationCurveType aType) {
	if (aType == mType) return;

	// save control points
	std::vector<glm::vec3> oldPts = interpolator().control_points();
	mType = aType;
	interpolator().set_control_points(oldPts);
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

void InterpolationCurve::set_control_points(std::vector<glm::vec3> pControlPoints) { interpolator().set_control_points(pControlPoints); }
const std::vector<glm::vec3>& InterpolationCurve::control_points() { return interpolator().control_points(); }
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
