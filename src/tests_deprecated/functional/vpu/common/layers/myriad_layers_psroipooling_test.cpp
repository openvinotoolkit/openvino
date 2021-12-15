// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_psroipooling_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayersTestsPSROIPooling_smoke,
        ::testing::Combine(
        ::testing::ValuesIn(s_PSROIPoolingLayerInput),
        ::testing::ValuesIn(s_PSROIPoolingLayerParam),
        ::testing::ValuesIn(s_PSROIPoolingNumROIs)));
