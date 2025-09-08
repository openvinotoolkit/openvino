// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/experimental_detectron_roifeatureextractor.hpp"

namespace {
using ov::test::ExperimentalDetectronROIFeatureExtractorLayerTest;

const std::vector<int64_t> outputSize = {7, 14};
const std::vector<int64_t> samplingRatio = {1, 2, 3};

const std::vector<std::vector<int64_t>> pyramidScales = {
        {8, 16, 32, 64},
        {4, 8, 16, 32},
        {2, 4, 8, 16}
};

const std::vector<std::vector<ov::test::InputShape>> staticInputShape = {
        ov::test::static_shapes_to_test_representation({{1000, 4}, {1, 8, 200, 336}, {1, 8, 100, 168}, {1, 8, 50, 84}, {1, 8, 25, 42}}),
        ov::test::static_shapes_to_test_representation({{1000, 4}, {1, 16, 200, 336}, {1, 16, 100, 168}, {1, 16, 50, 84}, {1, 16, 25, 42}}),
        ov::test::static_shapes_to_test_representation({{1200, 4}, {1, 8, 200, 42}, {1, 8, 100, 336}, {1, 8, 50, 168}, {1, 8, 25, 84}})
};

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalROI_static, ExperimentalDetectronROIFeatureExtractorLayerTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(staticInputShape),
                                 ::testing::ValuesIn(outputSize),
                                 ::testing::ValuesIn(samplingRatio),
                                 ::testing::ValuesIn(pyramidScales),
                                 ::testing::Values(false),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ExperimentalDetectronROIFeatureExtractorLayerTest::getTestCaseName);
} // namespace
