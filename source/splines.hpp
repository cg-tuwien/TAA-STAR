#pragma once

#include <glm/glm.hpp>


	struct Spline {
		float cam_t_max;
		std::vector<glm::vec3> camP;

		bool use_arclen;
		bool calced_arclen;

		//int camP_mode; // 0: off, 1: set cam target, 2: set abs. cam pos, look at camLookat

	 //  // linear interpol. of camLookat
		//float camL_T0, camL_T1;
		//glm::vec3 camL_start, camL_end;
		//int camL_mode; // 0: off, 1: interpol. camLookat

		//glm::vec3 camLookat;

		Spline(float cam_t_max, std::vector<glm::vec3> path) : cam_t_max(cam_t_max), camP(path) {}

		float map_arclen_t(float spline_t);

		glm::vec3 getPos(float t);
		void calc_arclen();
	private:
		//struct ArclenInfo { float spline_t, arc_t; };
		//std::vector<ArclenInfo> mArclenInfo;
		std::vector<float> mSegLenNormalized;
	};

