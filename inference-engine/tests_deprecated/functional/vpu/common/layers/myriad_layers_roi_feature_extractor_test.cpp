// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_roi_feature_extractor_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsROIFeatureExtractor_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_ROIFeatureExtractorLayerInput),
        ::testing::ValuesIn(s_ROIFeatureExtractorLayerParam),
        ::testing::ValuesIn(s_ROIFeatureExtractorNumROIs))
);
