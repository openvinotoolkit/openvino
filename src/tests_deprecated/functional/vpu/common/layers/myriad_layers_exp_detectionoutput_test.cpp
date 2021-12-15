// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_exp_detectionoutput_test.hpp"

const int _NumRois = 1000;
const int _NumClasses = 81;
const int _MaxDetections = 100;

static const std::vector<SizeParams> s_sizeParams_list =
{
    { _NumRois, _NumClasses, _MaxDetections },
};

static const std::vector<ExpDetectionOutputParams> s_layerParams_list =
{
    {{ 10.0, 10.0, 5.0, 5.0 }, 4.135166645050049, 0.5, 0.05, _MaxDetections, _NumClasses, 2000, 0 },
};

INSTANTIATE_TEST_SUITE_P(accuracy, myriadTestsExpDetectionOutput_smoke,
    ::testing::Combine(
        ::testing::ValuesIn(s_sizeParams_list),
        ::testing::ValuesIn(s_layerParams_list))
);
