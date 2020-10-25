// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_interpolate_test.hpp"

// INSTANTIATE_TEST_CASE_P(
// accuracy, myriadInterpolateLayerTests_nightly,
// ::testing::Combine(
// ::testing::ValuesIn(s_InterpolateInput),
// ::testing::Values<Factor>(2.0f, 0.5f),
// ::testing::Values<Antialias>(false),
// ::testing::Values<HwOptimization>(false, true),
// ::testing::Values(""))
// );

//#ifdef VPU_HAS_CUSTOM_KERNELS

// INSTANTIATE_TEST_CASE_P(
// accuracy_custom, myriadInterpolateLayerTests_nightly,
// ::testing::Combine(
// ::testing::ValuesIn(s_InterpolateInput),
// ::testing::Values<Factor>(2.0f),
// ::testing::Values<Antialias>(false, true),
// ::testing::Values<HwOptimization>(false, true),
// ::testing::Values(s_CustomConfig[1]))
// );

// #endif
