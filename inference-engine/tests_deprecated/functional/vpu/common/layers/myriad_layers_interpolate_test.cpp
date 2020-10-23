// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_interpolate_test.hpp"

INSTANTIATE_TEST_CASE_P(
	accuracy, myriadInterpolateLayerTests_smoke,
	::testing::Combine(
		::testing::ValuesIn(s_InterpolateInput),
		::testing::Values<Antialias>(false),
		::testing::Values<nearestMode>(0),
		::testing::Values<shapeCalcMode>(1),
		::testing::Values<coordTransMode>(0),
		::testing::Values<InterpolateAxis>(0, 1),
		::testing::Values<InterpolateScales>(2.0f, 2.0f),
		::testing::Values<HwOptimization>(false, true),
		::testing::Values<layoutPreference>(vpu::LayoutPreference::ChannelMajor, vpu::LayoutPreference::ChannelMinor),
		::testing::Values(""))
);

// #ifdef VPU_HAS_CUSTOM_KERNELS

// INSTANTIATE_TEST_CASE_P(
// 	accuracy_custom, myriadInterpolateLayerTests_nightly,
// 	::testing::Combine(
// 		::testing::ValuesIn(s_InterpolateInput),
// 		::testing::Values<Factor>(2.0f),
// 		::testing::Values<Antialias>(false, true),
// 		::testing::Values<HwOptimization>(false, true),
// 		::testing::Values(s_CustomConfig[1]))
// );

// #endif
