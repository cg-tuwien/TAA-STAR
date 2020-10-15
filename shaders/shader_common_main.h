
// Uniform buffer containing camera matrices and user input:
// It is updated every frame.
#define UNIFORMDEF_MatricesAndUserInput uniform MatricesAndUserInput {												\
	/* view matrix as returned from quake_camera */																	\
	mat4 mViewMatrix;																								\
	/* projection matrix as returned from quake_camera */															\
	mat4 mProjMatrix;																								\
	/* transformation matrix which tranforms to camera's position */												\
	mat4 mCamPos;																									\
	/* x = tessellation factor, y = displacement strength, z = use lighting/show normals, w = alpha threshold */	\
	vec4 mUserInput;																								\
}

// "mLightsources" uniform buffer containing all the light source data:
#define UNIFORMDEF_LightsourceData uniform LightsourceData {															\
	/* x,y ... ambient light sources start and end indices; z,w ... directional light sources start and end indices */	\
	uvec4 mRangesAmbientDirectional;																					\
	/* x,y ... point light sources start and end indices; z,w ... spot light sources start and end indices */			\
	uvec4 mRangesPointSpot;																								\
	/* Contains all the data of all the active light sources */															\
	LightsourceGpuData mLightData[128];																					\
}

// The actual material buffer (of type MaterialGpuData):
#define BUFFERDEF_Material buffer Material {	\
	MaterialGpuData materials[];				\
}

// ----- structure definitions

struct LightsourceGpuData
{
	/** Color of the light source. */
	vec4 mColor;
	/** Direction of the light source. */
	vec4 mDirection;
	/** Position of the light source. */
	vec4 mPosition;
	/** Angles, where the individual elements contain the following data: [0] cosine of halve outer cone angle, [1] cosine of halve inner cone angle, [2] falloff, [3] unused */
	vec4 mAnglesFalloff;
	/* Light source attenuation, where the individual elements contain the following data: [0] constant attenuation factor, [1] linear attenuation factor, [2] quadratic attenuation factor, [3], unused */
	vec4 mAttenuation;
	/** General information about the light source, where the individual elements contain the following data:[0] type of the light source */
	ivec4 mInfo;
};

// Material data struct definition:
struct MaterialGpuData {
	vec4 mDiffuseReflectivity;
	vec4 mAmbientReflectivity;
	vec4 mSpecularReflectivity;
	vec4 mEmissiveColor;
	vec4 mTransparentColor;
	vec4 mReflectiveColor;
	vec4 mAlbedo;

	float mOpacity;
	float mBumpScaling;
	float mShininess;
	float mShininessStrength;

	float mRefractionIndex;
	float mReflectivity;
	float mMetallic;
	float mSmoothness;

	float mSheen;
	float mThickness;
	float mRoughness;
	float mAnisotropy;

	vec4 mAnisotropyRotation;
	vec4 mCustomData;

	int mDiffuseTexIndex;
	int mSpecularTexIndex;
	int mAmbientTexIndex;
	int mEmissiveTexIndex;
	int mHeightTexIndex;
	int mNormalsTexIndex;
	int mShininessTexIndex;
	int mOpacityTexIndex;
	int mDisplacementTexIndex;
	int mReflectionTexIndex;
	int mLightmapTexIndex;
	int mExtraTexIndex;

	vec4 mDiffuseTexOffsetTiling;
	vec4 mSpecularTexOffsetTiling;
	vec4 mAmbientTexOffsetTiling;
	vec4 mEmissiveTexOffsetTiling;
	vec4 mHeightTexOffsetTiling;
	vec4 mNormalsTexOffsetTiling;
	vec4 mShininessTexOffsetTiling;
	vec4 mOpacityTexOffsetTiling;
	vec4 mDisplacementTexOffsetTiling;
	vec4 mReflectionTexOffsetTiling;
	vec4 mLightmapTexOffsetTiling;
	vec4 mExtraTexOffsetTiling;
};
