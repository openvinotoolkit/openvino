// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_resample_test.hpp"

// #-31522
INSTANTIATE_TEST_CASE_P(
	DISABLED_accuracy, myriadResampleLayerTests_smoke,
	::testing::Combine(
		::testing::ValuesIn(s_ResampleInput),
		::testing::Values<Factor>(2.0f, 0.5f),
		::testing::Values<Antialias>(false, true),
		::testing::Values<HwOptimization>(false, true),
		::testing::ValuesIn(s_CustomConfig))
);
