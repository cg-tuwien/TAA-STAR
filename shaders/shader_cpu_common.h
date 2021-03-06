#ifndef SHADER_CPU_COMMON_INCLUDED
#define SHADER_CPU_COMMON_INCLUDED 1

// specialization constants
#define SPECCONST_ID_TRANSPARENCY	    1u
#define SPECCONST_VAL_OPAQUE			0u
#define SPECCONST_VAL_TRANSPARENT		1u

// use a shadowmap?
#define ENABLE_SHADOWMAP 1
#define SHADOWMAP_SIZE 2048
#define SHADOWMAP_INITIAL_CASCADES 2

#define SHADOWMAP_MAX_CASCADES 4	// don't touch! Need to change uniform buffers and shaders too to increase that beyond 4

#if SHADOWMAP_INITIAL_CASCADES > SHADOWMAP_MAX_CASCADES
#error "SHADOWMAP_INITIAL_CASCADES > SHADOWMAP_MAX_CASCADES"
#endif

// Percentage of how much of a lightsource is removed when it is shadowed
#define SHADOW_OPACITY 0.9f
//#define SHADOW_OPACITY 1.0f

// enable raytracing?
#define ENABLE_RAYTRACING 1
#define RAYTRACING_MAX_SAMPLES_PER_PIXEL 32
#define RAYTRACING_CULLMASK_OPAQUE      0x01
#define RAYTRACING_CULLMASK_TRANSPARENT 0x02
#define RAYTRACING_APPROXIMATE_LOD 1

#define TAA_RTFLAG_OUT	0x0001
#define TAA_RTFLAG_DIS	0x0002
#define TAA_RTFLAG_NRM	0x0004
#define TAA_RTFLAG_DPT	0x0008
#define TAA_RTFLAG_MID	0x0010
#define TAA_RTFLAG_LUM	0x0020
#define TAA_RTFLAG_CNT	0x0040
#define TAA_RTFLAG_ALL	0x0080
#define TAA_RTFLAG_FXD	0x0100
#define TAA_RTFLAG_FXA	0x0200

// max. bones for animations
#define MAX_BONES	114

// GPU frustum culling
#define ENABLE_GPU_FRUSTUM_CULLING 1
#define GPU_FRUSTUM_CULLING_WORKGROUP_SIZE 32	// TODO: Test!

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

#define TAA_IMAGE_FORMAT_SEGMASK		vk::Format::eR32Uint
#define TAA_SHADER_FORMAT_SEGMASK		r32ui


#define	IMAGE_FORMAT_COLOR				vk::Format::eR16G16B16A16Sfloat
#define	IMAGE_FORMAT_DEPTH				vk::Format::eD32Sfloat
#define	IMAGE_FORMAT_NORMAL				vk::Format::eR32G32B32A32Sfloat
#define	IMAGE_FORMAT_MATERIAL			vk::Format::eR32Uint
#define IMAGE_FORMAT_VELOCITY			vk::Format::eR16G16B16A16Sfloat

#define IMAGE_FORMAT_SHADOWMAP			vk::Format::eD32Sfloat
#define SHADOWMAP_BINDING_SET			0
#define SHADOWMAP_BINDING_SLOT			6

#define	IMAGE_FORMAT_RAYTRACE			vk::Format::eR16G16B16A16Sfloat
#define SHADER_FORMAT_RAYTRACE			rgba16f

// for debugging
#define USE_DEBUG_POSBUFFERS	1
#define IMAGE_FORMAT_POSITION			vk::Format::eR16G16B16A16Sfloat

#endif
