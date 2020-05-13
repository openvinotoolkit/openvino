// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_roi_align_test.hpp"

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsROIAlign_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_ROIAlignLayerInput),
        ::testing::ValuesIn(s_ROIAlignLayerParam),
        ::testing::ValuesIn(s_ROIAlignNumROIs),
        ::testing::ValuesIn(s_ROIAlignMode)),
);
