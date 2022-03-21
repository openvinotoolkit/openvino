// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/experimental_detectron_roifeatureextractor.hpp"

using namespace ov::test;
using namespace ov::test::subgraph;

namespace {

const std::vector<int64_t> outputSize = {7};
const std::vector<int64_t> samplingRatio = {2};

const std::vector<std::vector<int64_t>> pyramidScales = {
        {1, 2, 4, 8},
};

const std::vector<std::vector<InputShape>> staticInputShape = {
        static_shapes_to_test_representation({{50, 4}, {1, 256, 160, 160}}),
};

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalROI_static, ExperimentalDetectronROIFeatureExtractorLayerTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(staticInputShape),
                                 ::testing::ValuesIn(outputSize),
                                 ::testing::ValuesIn(samplingRatio),
                                 ::testing::ValuesIn(pyramidScales),
                                 ::testing::Values(false),
                                 ::testing::Values(ov::element::Type_t::f16),
                                 ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                         ExperimentalDetectronROIFeatureExtractorLayerTest::getTestCaseName);
} // namespace
