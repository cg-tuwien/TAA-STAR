
// is the output from taa.comp already converted to sRGB-values?
#define TAA_OUTPUT_IS_SRGB	0

#if TAA_OUTPUT_IS_SRGB

// ye olde way
#define TAA_IMAGE_FORMAT_RGB		vk::Format::eR8G8B8A8Unorm
#define TAA_IMAGE_FORMAT_SRGB		vk::Format::eR8G8B8A8Srgb
#define TAA_SHADER_OUTPUT_FORMAT	r16f	// no visible difference when using rgba16f instead

// problem: with very low alpha values, ghosts linger in history forever

#else

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

#endif

#define TAA_IMAGE_FORMAT_POSTPROCESS	vk::Format::eR16G16B16A16Sfloat
#define TAA_SHADER_FORMAT_POSTPROCESS	rgba16f


#define	IMAGE_FORMAT_COLOR				vk::Format::eR16G16B16A16Sfloat
#define	IMAGE_FORMAT_DEPTH				vk::Format::eD32Sfloat
#define	IMAGE_FORMAT_NORMAL				vk::Format::eR32G32B32A32Sfloat
#define	IMAGE_FORMAT_MATERIAL			vk::Format::eR32Uint
#define IMAGE_FORMAT_VELOCITY			vk::Format::eR16G16B16A16Sfloat
