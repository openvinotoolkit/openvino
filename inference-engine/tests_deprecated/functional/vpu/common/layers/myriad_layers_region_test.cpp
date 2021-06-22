// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_region_test.hpp"

INSTANTIATE_TEST_SUITE_P(
	accuracy, myriadLayersTestsRegionYolo_smoke,
	::testing::Combine(
		::testing::Values<Coords>(4),
		::testing::Values<Classes>(20, 80),
		::testing::Values<Num>(5, 10),
		::testing::Values<MaskSize>(3),
		::testing::Values<DoSoftmax>(1, 0),
		::testing::Values(vpu::LayoutPreference::ChannelMajor, vpu::LayoutPreference::ChannelMinor),
		::testing::Values(IRVersion::v7, IRVersion::v10),
		::testing::Values("")
));

#ifdef VPU_HAS_CUSTOM_KERNELS

INSTANTIATE_TEST_SUITE_P(
	accuracy_custom, myriadLayersTestsRegionYolo_smoke,
	::testing::Combine(
		::testing::Values<Coords>(4),
		::testing::Values<Classes>(20),
		::testing::Values<Num>(5, 10),
		::testing::Values<MaskSize>(3),
		::testing::Values<DoSoftmax>(1, 0),
		::testing::Values(vpu::LayoutPreference::ChannelMajor, vpu::LayoutPreference::ChannelMinor),
		::testing::Values(IRVersion::v7, IRVersion::v10),
		::testing::Values(s_CustomConfig[1])
));

#endif
