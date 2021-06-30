// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_reorg_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsReorg_smoke, ::testing::Combine(
    ::testing::ValuesIn(s_ReorgInputs),
    ::testing::Values<Stride>(2),
    ::testing::Values(vpu::LayoutPreference::ChannelMinor, vpu::LayoutPreference::ChannelMajor),
    ::testing::Values(IRVersion::v7, IRVersion::v10),
    ::testing::Values<CustomConfig>({})
));

#ifdef VPU_HAS_CUSTOM_KERNELS

INSTANTIATE_TEST_SUITE_P(accuracy_custom, myriadLayersTestsReorg_smoke, ::testing::Combine(
    ::testing::ValuesIn(s_ReorgInputs_CustomLayer),
    ::testing::Values<Stride>(2),
    ::testing::Values(vpu::LayoutPreference::ChannelMinor, vpu::LayoutPreference::ChannelMajor),
    ::testing::Values(IRVersion::v7, IRVersion::v10),
    ::testing::Values(s_CustomConfig[1])
));

#endif
