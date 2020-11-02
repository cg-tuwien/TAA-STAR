// specialization constants
#define SPECCONST_ID_TRANSPARENCY	    1u
#define SPECCONST_VAL_OPAQUE			0u
#define SPECCONST_VAL_TRANSPARENT		1u

#define MAX_BONES	114

// don't have transparent movers (yet)

// 8-bit unorm - ugly! (interestingly: way worse than with explicit sRGB output)
//#define TAA_IMAGE_FORMAT_RGB		vk::Format::eR8G8B8A8Unorm
//#define TAA_SHADER_OUTPUT_FORMAT	rgba16f

// 32-bit float works *much* better
//#define TAA_IMAGE_FORMAT_RGB		vk::Format::eR32G32B32A32Sfloat
//#define TAA_SHADER_OUTPUT_FORMAT	rgba32f

// 16-bit float is fine too
#define TAA_IMAGE_FORMAT_RGB		vk::Format::eR16G16B16A16Sfloat
#define TAA_SHADER_OUTPUT_FORMAT	rgba16f

#define TAA_IMAGE_FORMAT_SRGB		vk::Format::eUndefined			// unused


#define TAA_IMAGE_FORMAT_POSTPROCESS	vk::Format::eR16G16B16A16Sfloat
#define TAA_SHADER_FORMAT_POSTPROCESS	rgba16f


#define	IMAGE_FORMAT_COLOR				vk::Format::eR16G16B16A16Sfloat
#define	IMAGE_FORMAT_DEPTH				vk::Format::eD32Sfloat
#define	IMAGE_FORMAT_NORMAL				vk::Format::eR32G32B32A32Sfloat
#define	IMAGE_FORMAT_MATERIAL			vk::Format::eR32Uint
#define IMAGE_FORMAT_VELOCITY			vk::Format::eR16G16B16A16Sfloat

// for debugging
#define USE_DEBUG_POSBUFFERS	1
#define IMAGE_FORMAT_POSITION			vk::Format::eR16G16B16A16Sfloat