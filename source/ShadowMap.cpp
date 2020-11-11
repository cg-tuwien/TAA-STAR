#include "ShadowMap.hpp"

void ShadowMap::init(const BoundingBox & aSceneBoundingBox, float camNear, float camFar, int aShadowMapTextureSize, int numCascades, bool autoCalcCascades)
{
	assert(numCascades > 0 && numCascades <= MAX_CASCADES);

	mSceneBoundingBox = aSceneBoundingBox;
	mCamNear = camNear;
	mCamFar  = camFar;
	mNumCascades = numCascades;
	mTextureSize = aShadowMapTextureSize;

	for (int i = 0; i < MAX_CASCADES; i++) {
		mCascadeProjMatrix[i] = mCascadeVPMatrix[i] = glm::mat4(1);
	}

	if (autoCalcCascades) calc_cascade_ends();
}

void ShadowMap::calc_cascade_ends() {
	// see article "Cascaded Shadow Maps" by Rouslan Dimitrov for NVIDIA OpenGL SDK sample about CSM
	float lambda = .75f;
	float ratio = mCamFar / mCamNear;
	for (int i = 1; i < mNumCascades; i++) {
		float z = lambda * mCamNear * powf(mCamFar / mCamNear, (float)i / mNumCascades) + (1.0f - lambda)*(mCamNear + ((float)i / mNumCascades)*(mCamFar - mCamNear));
		cascadeEnd[i - 1] = (z - mCamNear) / (mCamFar - mCamNear);
		//PRINTOUT("z(" << i << ")=" << z << ", cascadeEnd[" << i-1 << "]=" << cascadeEnd[i-1]);
	}
	cascadeEnd[mNumCascades - 1] = 1.0f;
}

void ShadowMap::calc(const glm::vec3 & aLightDirection, const glm::mat4 & aCamViewMatrix, const glm::mat4 & aCamProjMatrix, std::optional<glm::vec3> aIncludeThisPoint)
{
	mCamViewMatrix = aCamViewMatrix;
	mCamProjMatrix = aCamProjMatrix;
	mCamVPMatrix = mCamProjMatrix * mCamViewMatrix;
	mLightDirection = glm::normalize(aLightDirection);
	mAdjustForAdditionalPoint = aIncludeThisPoint;
	calcLightView();
}

void ShadowMap::calcLightView()
{
	// create light view matrix, centered at origin
	glm::vec3 lookFrom = glm::vec3(0);
	glm::vec3 lookAt   = mLightDirection;

	glm::vec3 up = glm::vec3(0,1,0);
	if (abs(glm::dot(up, mLightDirection)) > 0.999f) up = glm::vec3(0,0,-1);
	glm::vec3 right = glm::cross(mLightDirection, up);
	up = glm::cross(right, mLightDirection);
	mViewMatrix =  glm::lookAt(lookFrom, lookAt, up);

	// convert scene bounds to light space
	BoundingBox bbScene = mSceneBoundingBox;
	if (mAdjustForAdditionalPoint.has_value()) bbScene.combineWith(glm::vec4(mAdjustForAdditionalPoint.value(), 1.0f));
	glm::vec4 scenePtLS[8];
	bbScene.getTransformedPointsV4(mViewMatrix, scenePtLS);

	// calculate light projection
	for (int iCasc = 0; iCasc < mNumCascades; iCasc++) {
		// get the camera view frustum for the current cascade (in world space)
		glm::vec4 camFrustPtWS[8];
		float cascBegin;
		if (cascadeFitMode == CascadeFitMode::fitCascade) {
			cascBegin = (iCasc == 0) ? 0.0f : cascadeEnd[iCasc - 1];	// new cascade begins where previous cascade ended
		} else {
			cascBegin = 0.0f;	// all cascades start at zero
		}
		calcPartialCamFrustum(cascBegin, cascadeEnd[iCasc], camFrustPtWS);

		// convert cam frustum points from world space to light space
		glm::vec4 camFrustPtLS[8];
		for (int i = 0; i < 8; i++)
			camFrustPtLS[i] = mViewMatrix * camFrustPtWS[i];


		// TODO....

		BoundingBox bb;
		bb.calcFromPoints(8, camFrustPtLS);

		if (restrictLightViewToScene) {	// TODO: better way than intersecting with the bb(!) of the frustum?
			BoundingBox bb2;
			bb2.calcFromPoints(8, scenePtLS);
			bb.intersectWith(bb2);
		}

		// TODO....

		if (cascadeFitMode == CascadeFitMode::fitCascade) {
			if (texelSnapping) {
				glm::vec2 bb_min_xy = glm::vec2(bb.min);
				glm::vec2 bb_max_xy = glm::vec2(bb.max);
				glm::vec2 unitsPerTexel = (bb_max_xy - bb_min_xy) / (float)mTextureSize;
				bb_min_xy = glm::floor(bb_min_xy / unitsPerTexel) * unitsPerTexel;
				bb_max_xy = glm::floor(bb_max_xy / unitsPerTexel) * unitsPerTexel;
				bb.min.x = bb_min_xy.x; bb.min.y = bb_min_xy.y;
				bb.max.x = bb_max_xy.x; bb.max.y = bb_max_xy.y;
			}
		} // TODO: snapping for other modes

		// --- light ortho sides are fixed now (bb x and y values)


		// --- now fit the near and far plane

		float nearPlane, farPlane;
		if (nearfarFitMode == NearFarFitMode::nffFrustumOnly) {
			nearPlane = -bb.max.z;
			farPlane  = -bb.min.z;
		} else {
			// intersect scene box with light frustum -> gets tighter fit
			calcNearFar(glm::vec2(bb.min), glm::vec2(bb.max), /* out */ nearPlane, /* out */ farPlane, scenePtLS);
			// TODO... implement pancaking?
		}

		// create light projection matrix
		mCascadeProjMatrix[iCasc] = glm::ortho(bb.min.x, bb.max.x, bb.min.y, bb.max.y, nearPlane, farPlane);
		mCascadeVPMatrix[iCasc] = mCascadeProjMatrix[iCasc] * mViewMatrix;

		// calc cascade depth bounds	- TODO: move this out further (init?); when does cam proj change? on screen resize for instance..
		// TODO: improve performance!
		// ATTN: after projection z (when inside near/far) is also in NDC (-1..1), NOT in (0..1) !	// TODO - recheck for Vulkan!
		//glm::vec4 p = mCamProjMatrix * glm::vec4(0.0f, 0.0f, -mCamNear - cascadeEnd[iCasc] * (mCamFar - mCamNear), 1.0f);
		//mCascadeDepthBounds[iCasc] = (p.z / p.w) * .5f + .5f;

		// Vulkan:
		glm::vec4 p = mCamProjMatrix * glm::vec4(0.0f, 0.0f, mCamNear + cascadeEnd[iCasc] * (mCamFar - mCamNear), 1.0f);
		mCascadeDepthBounds[iCasc] = (p.z / p.w);
	}


}

void ShadowMap::calcNearFar(glm::vec2 lightMin, glm::vec2 lightMax, float &out_near, float &out_far, glm::vec4 *scenePtsLS) {
	// based on: https://github.com/walbourn/directx-sdk-samples/blob/master/CascadedShadowMaps11/CascadedShadowsManager.cpp
	// see also: https://docs.microsoft.com/en-us/windows/win32/dxtecharts/common-techniques-to-improve-shadow-depth-maps

	// scenePts (0-3 near, 4-7 far):
	//   6-------------7    y
	//   |\           /|    ^
	//   | \         / |    |
	//   |  2-------3  |    +---> x
	//   |  |       |  |
	//   |  |       |  |    (sketch is in world space with perspective, but passed points are in light space)
	//   |  0-------1  |    (NOTE: point order is different than in linked resources!)
	//   | /         \ |
	//   |/           \|
	//   4-------------5

	struct Triangle {
		glm::vec3 pt[3];
		bool culled;
	};

	// tessellation of the frustum:
	const static int sceneTriIdx[3 * 12] = {	// (winding order is irrelevant)
		0,1,2,	1,3,2,		// front
		5,4,7,  4,6,7,		// back
		2,3,6,	3,7,6,		// top
		0,1,4,	1,5,4,		// bottom
		0,4,2,	4,6,2,		// left
		5,1,7,	1,3,7		// right
	};

	out_near =  FLT_MAX;
	out_far  = -FLT_MAX;

	// iterate over all 12 triangles of the frustum
	for (int iFrustumTri = 0; iFrustumTri < 12; iFrustumTri++) {
		const int TRIANGLE_LIST_SIZE = 16;
		Triangle triangleList[TRIANGLE_LIST_SIZE];
		int triangleCount = 1;

		triangleList[0].pt[0] = scenePtsLS[sceneTriIdx[iFrustumTri * 3 + 0]];
		triangleList[0].pt[1] = scenePtsLS[sceneTriIdx[iFrustumTri * 3 + 1]];
		triangleList[0].pt[2] = scenePtsLS[sceneTriIdx[iFrustumTri * 3 + 2]];
		triangleList[0].culled = false;

		// clip triangle with the 4 (side) planes of the light frustum (simple comparison in light space), create new tris when needed
		for (int iPlane = 0; iPlane < 4; iPlane++) { // 0=left,1=right,2=bottom,3=top
			float edge;
			int component;
			switch (iPlane) {
			case 0:		edge = lightMin.x; component = 0; break;
			case 1:		edge = lightMax.x; component = 0; break;
			case 2:		edge = lightMin.y; component = 1; break;
			case 3:		edge = lightMax.y; component = 1; break;
			}

			// process triangle list
			for (int iTri = 0; iTri < triangleCount; iTri++) {
				if (triangleList[iTri].culled) continue;

				// check all 3 points of the triangle against the current plane
				int numPtsInside = 0;
				bool pointIsInside[3];
				for (int iTriPt = 0; iTriPt < 3; iTriPt++) {
					switch (iPlane) {
					case 0: pointIsInside[iTriPt] = triangleList[iTri].pt[iTriPt].x > lightMin.x; break;
					case 1: pointIsInside[iTriPt] = triangleList[iTri].pt[iTriPt].x < lightMax.x; break;
					case 2: pointIsInside[iTriPt] = triangleList[iTri].pt[iTriPt].y > lightMin.y; break;
					case 3: pointIsInside[iTriPt] = triangleList[iTri].pt[iTriPt].y < lightMax.y; break;
					}
					if (pointIsInside[iTriPt]) numPtsInside++;
				}

				// move inside points to the start of the array
				if (pointIsInside[1] && !pointIsInside[0]) { glm::vec3 t = triangleList[iTri].pt[0]; triangleList[iTri].pt[0] = triangleList[iTri].pt[1]; triangleList[iTri].pt[1] = t; pointIsInside[0] = true; pointIsInside[1] = false; }
				if (pointIsInside[2] && !pointIsInside[1]) { glm::vec3 t = triangleList[iTri].pt[1]; triangleList[iTri].pt[1] = triangleList[iTri].pt[2]; triangleList[iTri].pt[2] = t; pointIsInside[1] = true; pointIsInside[2] = false; }
				if (pointIsInside[1] && !pointIsInside[0]) { glm::vec3 t = triangleList[iTri].pt[0]; triangleList[iTri].pt[0] = triangleList[iTri].pt[1]; triangleList[iTri].pt[1] = t; pointIsInside[0] = true; pointIsInside[1] = false; }

				if (numPtsInside == 0) {
					// all points outside -> cull triangle
					triangleList[iTri].culled = true;
				} else if (numPtsInside == 1) {
					// one point inside (0) -> clip triangle
					glm::vec3 v01 = triangleList[iTri].pt[1] - triangleList[iTri].pt[0];
					glm::vec3 v02 = triangleList[iTri].pt[2] - triangleList[iTri].pt[0];
					float d = edge - triangleList[iTri].pt[0][component];	// distance from inside point to edge (sign is ok, cancels out in the next two lines)
					triangleList[iTri].pt[1] = triangleList[iTri].pt[0] + v01 * d / v01[component];
					triangleList[iTri].pt[2] = triangleList[iTri].pt[0] + v02 * d / v02[component];
					triangleList[iTri].culled = false;
				} else if (numPtsInside == 2) {
					// two points inside (0,1) -> cut into two triangles
					// move next tri out of the way (to end of list) to make space for the new tri
					assert(triangleCount < TRIANGLE_LIST_SIZE);
					triangleList[triangleCount] = triangleList[iTri + 1];

					glm::vec3 v20 = triangleList[iTri].pt[0] - triangleList[iTri].pt[2];
					glm::vec3 v21 = triangleList[iTri].pt[1] - triangleList[iTri].pt[2];
					float d = edge - triangleList[iTri].pt[2][component];	// distance from outside point to edge (sign is ok, cancels out in the next lines)

																			// new triangle (0, 1, point on 2-0-edge; inserted to list)
					triangleList[iTri + 1].pt[0] = triangleList[iTri].pt[0];
					triangleList[iTri + 1].pt[1] = triangleList[iTri].pt[1];
					triangleList[iTri + 1].pt[2] = triangleList[iTri].pt[2] + v20 * d / v20[component];
					triangleList[iTri + 1].culled = false;

					// new triangle (1, point on 2-0-edge, point on 2-1-edge; overwrites old tri in list)
					triangleList[iTri].pt[0] = triangleList[iTri + 1].pt[1];
					triangleList[iTri].pt[1] = triangleList[iTri + 1].pt[2];
					triangleList[iTri].pt[2] = triangleList[iTri].pt[2] + v21 * d / v21[component];
					triangleList[iTri].culled = false;

					triangleCount++; // one more triangle
					iTri++; // but skip it now
				} else {
					// all points inside
					triangleList[iTri].culled = false;
				}
			}

		}

		// update near/far plane from the triangles resulting from this frustum triangle
		for (int iTri = 0; iTri < triangleCount; iTri++) {
			if (triangleList[iTri].culled) continue;
			for (int iTriPt = 0; iTriPt < 3; iTriPt++) {
				float z = -triangleList[iTri].pt[iTriPt].z;
				if (out_near > z) out_near = z;
				if (out_far  < z) out_far  = z;
			}
		}

	}

}


// calc camera view frustum corner points in world space
void ShadowMap::getCamFrustum(glm::vec4 * out_frustumPoints) {
	// order: near bottom left, near bottom right, near top left, near top right, 
	//        far  bottom left, far  bottom right, far  top left, far  top right
	glm::mat4 invCamVP = glm::inverse(mCamVPMatrix);
	int i = 0;
	for (int z = 0; z < 2; z++) {
		for (int y = 0; y < 2; y++) {
			for (int x = 0; x < 2; x++) {
				//glm::vec4 p(x * 2.0f - 1.0f, y * 2.0f - 1.0f, z * 2.0f - 1.0f, 1.0f);
				glm::vec4 p(x * 2.0f - 1.0f, y * 2.0f - 1.0f, z, 1.0f);
				p = invCamVP * p;
				out_frustumPoints[i++] = p / p.w;
			}
		}
	}
}

// calculate corner points of a sub-frustum of the current camera view frustum
void ShadowMap::calcPartialCamFrustum(float beginFactor, float endFactor, glm::vec4 * out_frustumPoints) {
	glm::vec4 pts[8];
	getCamFrustum(pts);
	for (int i = 0; i < 4; i++) {
		glm::vec4 v = pts[i + 4] - pts[i];
		out_frustumPoints[i]   = pts[i] + v * beginFactor;
		out_frustumPoints[i+4] = pts[i] + v * endFactor;
	}
}
