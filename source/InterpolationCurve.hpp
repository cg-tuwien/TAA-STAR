#pragma once


enum class InterpolationCurveType {bezier_curve, quadratic_uniform_b_spline, cubic_uniform_b_spline, catmull_rom_spline};

class InterpolationCurve {
public:
	InterpolationCurve() : mType(InterpolationCurveType::bezier_curve) {}
	InterpolationCurve(InterpolationCurveType aType, std::vector<glm::vec3> pControlPoints) : mType(aType) { interpolator().set_control_points(pControlPoints); }

	void setType(InterpolationCurveType aType);
	InterpolationCurveType type() { return mType; }

	void set_control_points(std::vector<glm::vec3> pControlPoints);
	const std::vector<glm::vec3>& control_points();
	size_t num_control_points();
	glm::vec3 value_at(float t);
	glm::vec3 slope_at(float t);

	float mapConstantSpeedTime(float t);

	int  arcLenSamplesPerSegment() { return mArcLenSamplesPerSegment; }
	void setArcLenSamplesPerSegment(int numSamples) { mArcLenSamplesPerSegment = numSamples; invalidateArcLenTable(); }

	bool valid();
private:
	InterpolationCurveType mType;

	gvk::bezier_curve				mInterpolatorBezier;
	gvk::quadratic_uniform_b_spline	mInterpolatorQuadB;
	gvk::cubic_uniform_b_spline		mInterpolatorCubeB;
	gvk::catmull_rom_spline			mInterpolatorCatmull;

	int mArcLenSamplesPerSegment = 20;

	std::vector<float> mArcLenTable;

	gvk::cp_interpolation & interpolator();
	void buildArcLenTable();
	void invalidateArcLenTable();
};

