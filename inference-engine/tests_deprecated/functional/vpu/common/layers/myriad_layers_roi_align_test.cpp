// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_roi_align_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsROIAlign_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_ROIAlignLayerInput),
        ::testing::ValuesIn(s_ROIAlignLayerParam),
        ::testing::ValuesIn(s_ROIAlignNumROIs),
        ::testing::ValuesIn(s_ROIAlignMode))
);

INSTANTIATE_TEST_SUITE_P(accuracy_faster, myriadLayersTestsROIAlign_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_ROIAlignLayerInput_Faster),
        ::testing::ValuesIn(s_ROIAlignLayerParam_Faster),
        ::testing::ValuesIn(s_ROIAlignNumROIs_Faster),
        ::testing::ValuesIn(s_ROIAlignMode_Faster))
);
