// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_reorg_test.hpp"

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsReorg_smoke, ::testing::Combine(
    ::testing::ValuesIn(s_ReorgInputs),
    ::testing::Values<Stride>(2),
    ::testing::Values(vpu::LayoutPreference::ChannelMinor, vpu::LayoutPreference::ChannelMajor),
    ::testing::Values(IRVersion::v7, IRVersion::v10),
    ::testing::ValuesIn(s_CustomConfig)
));
