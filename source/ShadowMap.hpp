#pragma once

#include "BoundingBox.hpp"

class ShadowMap {
public:
	static const int MAX_CASCADES = 4;
private:
	glm::mat4 mViewMatrix;
	int mNumCascades;
	float mCamNear, mCamFar;
	int mTextureSize;

	glm::mat4 mCascadeProjMatrix[MAX_CASCADES];
	glm::mat4 mCascadeVPMatrix[MAX_CASCADES];
	float mCascadeDepthBounds[MAX_CASCADES];

	BoundingBox mSceneBoundingBox;

	std::optional<glm::vec3> mAdjustForAdditionalPoint;

	glm::mat4 mCamViewMatrix, mCamProjMatrix, mCamVPMatrix;
	glm::vec3 mLightDirection;

	void calcLightView();
	void calcNearFar(glm::vec2 lightMin, glm::vec2 lightMax, float &out_near, float &out_far, glm::vec4 *scenePtsLS);
	void getCamFrustum(glm::vec4 *out_frustumPoints);
	void calcPartialCamFrustum(float beginFactor, float endFactor, glm::vec4 *out_frustumPoints);

public:
	enum class CascadeFitMode {fitCascade, fitScene};
	enum class NearFarFitMode {nffFrustumOnly, nffIntersect, /* nffIntersectAndPancake */};

	// params
	float cascadeEnd[MAX_CASCADES] = {0.005f, 0.02f, 0.10f, 1.0f};	// only used if autoCalcCascades is false in init()
	CascadeFitMode cascadeFitMode = CascadeFitMode::fitCascade;
	bool restrictLightViewToScene = false; // clip frustum to scene? tighter bound, but other problems... not generally recommended by resources
	NearFarFitMode nearfarFitMode = NearFarFitMode::nffIntersect;
	bool texelSnapping = true;

	void init(const BoundingBox &aSceneBoundingBox, float camNear, float camFar, int aShadowMapTextureSize, int numCascades, bool autoCalcCascades);
	void calc(const glm::vec3 &aLightDirection, const glm::mat4 &aCamViewMatrix, const glm::mat4 &aCamProjMatrix, std::optional<glm::vec3> aIncludeThisPoint = std::nullopt);
	void calc_cascade_ends();
	glm::mat4 view_matrix() { return mViewMatrix; }
	glm::mat4 projection_matrix(int cascade = 0) { return mCascadeProjMatrix[cascade]; }
	float max_depth(int cascade) { return mCascadeDepthBounds[cascade]; }
};
