#pragma once

#include <gvk.hpp>
#include <random>

namespace helpers
{
	// ac: Halton sequence, see https://en.wikipedia.org/wiki/Halton_sequence
	static float halton(int i, int b) { // index, base; (index >= 1)
		float f = 1.0f, r = 0.0f;
		while (i > 0) {
			f = f / b;
			r = r + f * (i % b);
			i = i / b;
		}
		return r;
	}

	template <size_t Len>
	static std::array<glm::vec2, Len> halton_2_3(glm::vec2 aScale) {
		std::array<glm::vec2, Len> result;
		for (size_t i = 0; i < Len; i++) {
			result[i] = aScale * glm::vec2{ halton(int(i) + 1, 2) - 0.5f, halton(int(i) + 1, 3) - 0.5f };
		}
		return result;
	}

	static void exclude_a_curtain(std::vector<std::tuple<avk::resource_reference<const gvk::model_t>, std::vector<size_t>>>& aSelectedModelsAndMeshes)
	{
		size_t a = 0;
		for (; a < aSelectedModelsAndMeshes.size(); ++a) {
			auto& tpl = aSelectedModelsAndMeshes[a];
			auto model = std::get<avk::resource_reference<const gvk::model_t>>(tpl);
			if (model->path().find("fabric") == std::string::npos) {
				continue;
			}
			std::vector<size_t>& meshIndices = std::get<std::vector<size_t>>(tpl);
			for (size_t i = 0; i < meshIndices.size(); ++i) {
				auto meshName = model->name_of_mesh(meshIndices[i]);
				if (meshName == "sponza_320") {
					meshIndices.erase(meshIndices.begin() + i);

					if (std::get<std::vector<size_t>>(aSelectedModelsAndMeshes[a]).empty()) {
						aSelectedModelsAndMeshes.erase(aSelectedModelsAndMeshes.begin() + a);
					}

					return;
				}
			}
		}
	}

	static void set_terrain_material_config(avk::resource_reference<gvk::orca_scene_t> aScene)
	{
		auto m = gvk::material_config{};
		m.mAmbientReflectivity	= glm::vec4{ 1.0f, 1.0f, 1.0f, 1.0f };
		m.mDiffuseReflectivity	= glm::vec4{ 1.0f, 1.0f, 1.0f, 1.0f };
		m.mSpecularReflectivity	= glm::vec4{ 0.0f, 0.0f, 0.0f, 0.0f };
		m.mEmissiveColor		= glm::vec4{ 0.0f, 0.0f, 0.0f, 0.0f };
		m.mShininess			= 100.0f;
		m.mDiffuseTex			= "assets/terrain/large_metal_debris_Base_Color.jpg";
		m.mNormalsTex			= "assets/terrain/large_metal_debris_Normal.jpg";
		m.mHeightTex			= "assets/terrain/large_metal_debris_Displacement.jpg";


		// Select the terrain models/meshes
		auto terrainModelIndices = aScene->select_models([](size_t index, const gvk::model_data& modelData){
			return std::string::npos != modelData.mName.find("terrain") || std::string::npos != modelData.mName.find("debris");
		});

		// Assign the material config to all meshes
		for (auto i : terrainModelIndices) {
			if (std::string::npos != aScene->model_at_index(i).mName.find("terrain")) {
				m.mDiffuseTexOffsetTiling	= glm::vec4{ 0.0f, 0.0f, 32.0f, 32.0f };
				m.mSpecularTexOffsetTiling	= glm::vec4{ 0.0f, 0.0f, 32.0f, 32.0f };
				m.mNormalsTexOffsetTiling	= glm::vec4{ 0.0f, 0.0f, 32.0f, 32.0f };
				m.mHeightTexOffsetTiling	= glm::vec4{ 0.0f, 0.0f, 32.0f, 32.0f };
			}
			else {
				m.mDiffuseTexOffsetTiling	= glm::vec4{ 0.0f, 0.0f, 10.0f, 10.0f };
				m.mSpecularTexOffsetTiling	= glm::vec4{ 0.0f, 0.0f, 10.0f, 10.0f };
				m.mNormalsTexOffsetTiling	= glm::vec4{ 0.0f, 0.0f, 10.0f, 10.0f };
				m.mHeightTexOffsetTiling	= glm::vec4{ 0.0f, 0.0f, 10.0f, 10.0f };
			}

			auto meshes = aScene->model_at_index(i).mLoadedModel->select_all_meshes();
			for (auto j : meshes) {
				aScene->model_at_index(i).mLoadedModel->set_material_config_for_mesh(j, m);
			}
		}
	}

	static void increase_specularity_of_some_submeshes(gvk::orca_scene_t& aScene)
	{
		// Select the terrain models/meshes
		auto sponzaStructureIndex = aScene.select_models([](size_t index, const gvk::model_data& modelData){
			return std::string::npos != modelData.mName.find("sponza_structure");
		});

		assert(sponzaStructureIndex.size() == 1);
		
		// Assign the material config to the "floor" and "lion" meshes
		auto& model = aScene.model_at_index(sponzaStructureIndex[0]).mLoadedModel;
		auto meshes = model->select_meshes([&model](size_t meshIndex, const aiMesh*){
			return 0 == model->name_of_mesh(meshIndex).find("floor") || 0 == model->name_of_mesh(meshIndex).find("lion");
		});
		for (auto j : meshes) {
			auto mat = model->material_config_for_mesh(j);
			mat.mReflectiveColor = glm::vec4{0.9f};
			mat.mCustomData[2] = 0.75f; // Set a normal mapping strength decrease factor
			model->set_material_config_for_mesh(j, mat);
			auto matafterarsch = model->material_config_for_mesh(j);
			auto matafterarsc2h = model->material_config_for_mesh(j);
			
		}
	}

	// We're only going to tessellate terrain materials. Set the tessellation factor for those to 1.
	// Indicate that the other materials shall not be tessellated/displaced with a tessellation factor of 0.
	static void enable_tessellation_for_specific_meshes(gvk::orca_scene_t& aScene)
	{
		for (auto& model : aScene.models()) {
			const bool isToBeTessellated = std::string::npos != model.mName.find("terrain") || std::string::npos != model.mName.find("debris");
			auto meshIndices = model.mLoadedModel->select_all_meshes();
			for (auto i : meshIndices) {
				auto m = model.mLoadedModel->material_config_for_mesh(i);
				m.mCustomData[0] = isToBeTessellated ? 1.0f : 0.0f;
				model.mLoadedModel->set_material_config_for_mesh(i, m);
			}
		}
	}

	// makes only sense for meshes that are to be tessellated
	static void set_mesh_specific_displacement_strength(gvk::orca_scene_t& aScene)
	{
		for (auto& model : aScene.models()) {
			// Compute a displacement strength that fits to the normal map strength:

			// Displacement distance in relation to the texel size - texture specific value
			auto displacementInTexels = 400.0f;

			// Average size of u and v in object space (actually, mesh specific value)
			auto uvScaleOS = 200.0f;
			if (std::string::npos != model.mName.find("terrain")) {
				uvScaleOS = 2040.0f;
			}
			if (std::string::npos != model.mName.find("debris")) {
				uvScaleOS = 1200.0f;
			}

			auto meshIndices = model.mLoadedModel->select_all_meshes();
			for (auto i : meshIndices) {
				auto m = model.mLoadedModel->material_config_for_mesh(i);
				const bool isToBeTessellated = m.mCustomData[0] != 0.0f;
				if (!isToBeTessellated) {
					continue;
				}

				// Compute approximate size of a texel in object space, which depends on the
				// average size of u and v in object space, the texture's size and tiling.

				int width = 1024, height = 1024, comp = 4; // just init with something if stbi_info fails
				stbi_info(m.mHeightTex.c_str(), &width, &height, &comp);

				auto tiling = m.mHeightTexOffsetTiling[2];

				auto texelSizeOS = uvScaleOS / (tiling * width);

				// Compute the displacement strength factor for this mesh in object space
				// (actually, transform m_displacement_strength from "texture space" to object space)
				float displacementStrengthFactorOS = displacementInTexels * texelSizeOS;

				m.mCustomData[1] = displacementStrengthFactorOS;
				model.mLoadedModel->set_material_config_for_mesh(i, m);
			}
		}
	}

	// Struct for quad vertex attributes:
	// (Also used for describint the vertex input attributes of assignment3::mPipelineLightingPass)
	struct quad_vertex {
		glm::vec3 mPosition;
		glm::vec2 mTextureCoordinate;
	};

	static std::tuple<avk::resource_reference<const avk::buffer_t>, avk::resource_reference<const avk::buffer_t>> get_quad_vertices_and_indices()
	{
		// Vertex data for a quad:
		static const std::vector<quad_vertex> sQuadVertices = {
			{{  1.0f,  1.0f, 0.0f }, { 1.0f, 1.0f }},
			{{ -1.0f,  1.0f, 0.0f }, { 0.0f, 1.0f }},
			{{ -1.0f, -1.0f, 0.0f }, { 0.0f, 0.0f }},
			{{  1.0f, -1.0f, 0.0f }, { 1.0f, 0.0f }},
		};

		// Indices for the quad:
		static const std::vector<uint16_t> sQuadIndices = {
			0, 1, 2,   2, 3, 0
		};

		// Create and upload vertex data for a quad
		static auto sQuadVertexBuffer = [&]() {
			auto b = gvk::context().create_buffer(
				avk::memory_usage::device, {},
				avk::vertex_buffer_meta::create_from_data(sQuadVertices)
			);
			b->fill(sQuadVertices.data(), 0, avk::sync::with_barriers(gvk::context().main_window()->command_buffer_lifetime_handler()));
			return b;
		}();

		// Create and upload incides for drawing the quad
		static auto sQuadIndexBuffer = [&]() {
			auto b = gvk::context().create_buffer(
				avk::memory_usage::device, {},
				avk::index_buffer_meta::create_from_data(sQuadIndices)
			);
			b->fill(sQuadIndices.data(), 0, avk::sync::with_barriers(gvk::context().main_window()->command_buffer_lifetime_handler()));
			return b;
		}();

		assert (sQuadVertexBuffer->has_meta<avk::vertex_buffer_meta>());
		assert (sQuadIndexBuffer->has_meta<avk::index_buffer_meta>());
		return std::forward_as_tuple(avk::const_referenced(sQuadVertexBuffer), avk::const_referenced(sQuadIndexBuffer));
	}

	static std::vector<gvk::lightsource>& get_lights()
	{
		static std::vector<gvk::lightsource> sLightsources = [](){
			std::vector<gvk::lightsource> ls;

			// Ambient light:
			ls.push_back(gvk::lightsource::create_ambient(glm::vec3{1.0f/255.0f, 2.0f/255.0f, 3.0f/255.0f} * 0.5f, "ambient light"));

			// Directional light:
			ls.push_back(gvk::lightsource::create_directional(glm::vec3(-0.38f, -0.78f, 0.0f), glm::vec3{13.0f/255.0f, 17.0f/255.0f, 27.0f/255.0f} * 4.0f, "directional light"));

			std::vector<glm::vec3> lightColors;
			lightColors.emplace_back(1.0f, 1.0f, 1.0f);
			lightColors.emplace_back(0.878f, 1.000f, 1.000f);
			lightColors.emplace_back(0.957f, 0.643f, 0.376f);
			lightColors.emplace_back(0.000f, 0.000f, 1.000f);
			lightColors.emplace_back(0.251f, 0.878f, 0.816f);
			lightColors.emplace_back(0.000f, 0.980f, 0.604f);
			lightColors.emplace_back(0.545f, 0.000f, 0.545f);
			lightColors.emplace_back(1.000f, 0.000f, 1.000f);
			lightColors.emplace_back(0.984f, 1.000f, 0.729f);
			lightColors.emplace_back(0.780f, 0.082f, 0.522f);
			lightColors.emplace_back(1.000f, 0.843f, 0.000f);
			lightColors.emplace_back(0.863f, 0.078f, 0.235f);
			lightColors.emplace_back(0.902f, 0.902f, 0.980f);
			lightColors.emplace_back(0.678f, 1.000f, 0.184f);

			std::default_random_engine generator;
			generator.seed(186);
			std::uniform_int_distribution<size_t> distribution(0, lightColors.size() - 1); // generates number in the range 0..light_colors.size()-1

			// Create a light near the walkthrough
			ls.push_back(gvk::lightsource::create_pointlight({-0.64f, 0.45f, 3.35f}, lightColors[distribution(generator)] * 3.0f, "pointlight near walkthrough").set_attenuation(1.0f, 0.0f, 5.0f));

			// Create a larger light outside above the terrain
			ls.push_back(gvk::lightsource::create_pointlight({-2.0f, 1.45f, 17.0f}, lightColors[distribution(generator)] * 3.0f, "pointlight outside above terrain").set_attenuation(1.0f, 0.0f, 1.2f));

			{ // Create lots of small lights near the floor
				const auto lbX = -14.2f;
				const auto lbZ = -6.37f;
				const auto nx = 13;
				const auto nz = 6;
				const auto stepX = (12.93f - lbX) / (nx - 1);
				const auto stepZ = (5.65f - lbZ) / (nz - 1);
				for (auto x = 0; x < nx; ++x) {
					for (auto z = 0; z < nz; ++z) {
						ls.push_back(gvk::lightsource::create_pointlight(glm::vec3(lbX + x * stepX, 0.1f, lbZ + z * stepZ), lightColors[distribution(generator)]).set_attenuation(1.0f, 0.0f, 30.0f));
					}
				}
			}

			{	// Create several larger lights near the ceiling
				const auto lbX = -13.36f;
				const auto lbZ = -5.46f;
				const auto nx = 6;
				const auto nz = 3;
				const auto stepX = (12.1f - lbX) / (nx - 1);
				const auto stepZ = (4.84f - lbZ) / (nz - 1);
				for (auto x = 0; x < nx; ++x) {
					for (auto z = 0; z < nz; ++z) {
						ls.push_back(gvk::lightsource::create_pointlight(glm::vec3(lbX + x * stepX, 7.0f, lbZ + z * stepZ), lightColors[distribution(generator)], fmt::format("pointlight[{}|{}]", x, z)).set_attenuation(1.0f, 0.0f, 5.666f));
					}
				}
			}

			// create a bigger spot light pointing to the corner
			ls.push_back(gvk::lightsource::create_spotlight({2.0f, 0.0f, 0.0f}, {1.0f, 0.2f, 0.5f}, glm::half_pi<float>(), 0.0f, 1.0f, glm::vec3{1.0f, 0.0f, 0.0f}, "big spotlight towards corner").set_attenuation(1.0f, 0.1f, 0.01f));

			// create spot lights in the arches
			generator.seed(186);
			for (auto i = 0; i < 5; ++i)
			{
				const auto direction = glm::vec3(0.f, -1.f, 0.f);
				ls.push_back(gvk::lightsource::create_spotlight(glm::vec3(-8.03f + i * 3.72f, 3.76f, -2.6f), direction, 1.08f, 0.99f, 1.0f, lightColors[distribution(generator)], fmt::format("spotlight[{}|here]", i)).set_attenuation(1.0f, 0.0f, .666f));
				ls.push_back(gvk::lightsource::create_spotlight(glm::vec3(-8.03f + i * 3.72f, 3.76f,  2.0f), direction, 1.08f, 0.99f, 1.0f, lightColors[distribution(generator)], fmt::format("spotlight[{}|there]", i)).set_attenuation(1.0f, 0.0f, .666f));
			}

			return ls;
		}();
		return sLightsources;
	}

	static void animate_lights(std::vector<gvk::lightsource>& aLightsources, float aElapsedTime)
	{
		{
			const auto it = std::find_if(std::begin(aLightsources), std::end(aLightsources), [](const gvk::lightsource& ls) { return "pointlight near walkthrough" == ls.mName; });
			if (std::end(aLightsources) != it) {
				const auto speedXZ = 0.5f;
				const auto radiusXZ = 1.5f;
				it->mPosition = glm::vec3{-0.64f, 0.45f, 3.35f} + glm::vec3(radiusXZ * glm::sin(speedXZ * aElapsedTime), 0.0f, radiusXZ * glm::cos(speedXZ * aElapsedTime));
			}
		}
		{
			const auto it = std::find_if(std::begin(aLightsources), std::end(aLightsources), [](const gvk::lightsource& ls) { return "pointlight outside above terrain" == ls.mName; });
			if (std::end(aLightsources) != it) {
				const auto speedXZ = 0.75f;
				const auto radiusXZ = 4.0f;
				it->mPosition = glm::vec3{-2.0f, 1.45f, 17.0f} + glm::vec3(radiusXZ * glm::sin(speedXZ * aElapsedTime), 0.0f, radiusXZ * glm::cos(speedXZ * aElapsedTime));
			}
		}
	}

	static uint32_t get_lightsource_type_begin_index(gvk::lightsource_type aLightsourceType)
	{
		auto lights = get_lights();
		auto it = std::lower_bound(std::begin(lights), std::end(lights), aLightsourceType, [](const gvk::lightsource& a, const gvk::lightsource_type& b){
			typedef std::underlying_type<gvk::lightsource_type>::type EnumType;
			return static_cast<EnumType>(a.mType) < static_cast<EnumType>(b);
		});
		return static_cast<uint32_t>(std::distance(std::begin(lights), it));
	}

	static uint32_t get_lightsource_type_end_index(gvk::lightsource_type aLightsourceType)
	{
		auto lights = get_lights();
		auto it = std::upper_bound(std::begin(lights), std::end(lights), aLightsourceType, [](const gvk::lightsource_type& a, const gvk::lightsource& b){
			typedef std::underlying_type<gvk::lightsource_type>::type EnumType;
			return static_cast<EnumType>(a) < static_cast<EnumType>(b.mType);
		});
		return static_cast<uint32_t>(std::distance(std::begin(lights), it));
	}

	static std::unordered_map<std::string, std::tuple<vk::UniqueQueryPool, std::array<uint32_t, 2>, float>> sIntervals;

	static vk::QueryPool& add_timing_interval_and_get_query_pool(const std::string& aName)
	{
		auto iter = sIntervals.find(aName);
		if (iter == sIntervals.end()) {
			vk::QueryPoolCreateInfo queryPoolCreateInfo;
			queryPoolCreateInfo.setQueryCount(2);
			queryPoolCreateInfo.setQueryType(vk::QueryType::eTimestamp);

			iter = sIntervals.try_emplace(aName, gvk::context().device().createQueryPoolUnique(queryPoolCreateInfo), std::array<uint32_t, 2>(), 0.0f).first;
		}
		return *std::get<0>(iter->second);
	}

	static void record_timing_interval_start(const vk::CommandBuffer& aCommandBuffer, const std::string& aName)
	{
		auto& queryPool = add_timing_interval_and_get_query_pool(aName);
		aCommandBuffer.resetQueryPool(queryPool, 0u, 2u);
		aCommandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eAllGraphics, queryPool, 0u);
	}

	static void record_timing_interval_end(const vk::CommandBuffer& aCommandBuffer, const std::string& aName)
	{
		auto& queryPool = add_timing_interval_and_get_query_pool(aName);
		aCommandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eAllGraphics, queryPool, 1u);
	}

	// request last timing interval from GPU and return averaged interval from previous measurements (in ms)
	static float get_timing_interval_in_ms(const std::string& aName)
	{
		auto iter = sIntervals.find(aName);
		if (iter == sIntervals.end()) {
			return 0.0f;
		}
		auto& [queryPool, timestamps, avgRendertime] = iter->second;
		gvk::context().device().getQueryPoolResults(*queryPool, 0u, 2u, sizeof(timestamps), timestamps.data(), sizeof(uint32_t), vk::QueryResultFlagBits::eWait);
		float delta = (timestamps[1] - timestamps[0]) * gvk::context().physical_device().getProperties().limits.timestampPeriod / 1000000.0f;
		avgRendertime = avgRendertime * 0.9f + delta * 0.1f;
		return avgRendertime;
	}

	static void clean_up_timing_resources()
	{
		sIntervals.clear();
	}
}
