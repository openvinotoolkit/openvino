// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_reorg_test.hpp"

static std::vector<std::string> s_CustomConfig = {
    "",
#ifdef VPU_HAS_CUSTOM_KERNELS
    getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"
#endif
};

static std::vector<layoutPreference> layoutPreferences = {
    vpu::LayoutPreference::ChannelMajor,
#ifndef VPU_HAS_CUSTOM_KERNELS
    vpu::LayoutPreference::ChannelMinor
#endif
};

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsReorg_nightly, ::testing::Combine(
    ::testing::Values<DimsInput>(
        MAKE_STRUCT(tensor_test_params, 1, 64, 26, 26),
        MAKE_STRUCT(tensor_test_params, 1, 192, 6 * 26, 6 * 26),
        MAKE_STRUCT(tensor_test_params, 1,  4,  6,  6)
    ),
    ::testing::Values<ScaleOutput>(2),
    ::testing::Values<Stride>(2),
    ::testing::ValuesIn(layoutPreferences),
    ::testing::ValuesIn(s_CustomConfig),
    ::testing::Values<IRVersion>(IRVersion::v7, IRVersion::v10)
));
