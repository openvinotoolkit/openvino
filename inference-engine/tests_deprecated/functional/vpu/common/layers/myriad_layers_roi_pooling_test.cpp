// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_roi_pooling_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsROIPooling_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_ROIPoolingLayerInput),
        ::testing::ValuesIn(s_ROIPoolingLayerParam),
        ::testing::ValuesIn(s_ROIPoolingNumRois),
        ::testing::ValuesIn(s_ROIPoolingMethod),
        ::testing::Values<IRVersion>(IRVersion::v7, IRVersion::v10))
);
