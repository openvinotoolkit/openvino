// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/experimental_detectron_prior_grid_generator.hpp"

namespace {
using ov::test::ExperimentalDetectronPriorGridGeneratorLayerTest;

std::vector<std::vector<ov::test::InputShape>> shapes = {
    ov::test::static_shapes_to_test_representation({{3, 4}, {1, 16, 4, 5}, {1, 3, 100, 200}}),
    ov::test::static_shapes_to_test_representation({{3, 4}, {1, 16, 3, 7}, {1, 3, 100, 200}}),
    {
        // priors
        {{-1, -1}, {{3, 4}, {3, 4}}},
        // feature_map
        {{-1, -1, -1, -1}, {{1, 16, 4, 5}, {1, 16, 100, 100}}},
        // im_data
        {{-1, -1, -1, -1}, {{1, 3, 100, 200}, {1, 3, 100, 200}}}
    },
    {
        // priors
        {{-1, -1}, {{3, 4}, {3, 4}}},
        // feature_map
        {{-1, -1, -1, -1}, {{1, 16, 3, 7}, {1, 16, 100, 100}}},
        // im_data
        {{-1, -1, -1, -1}, {{1, 3, 100, 200}, {1, 3, 100, 200}}}
    }
};

std::vector<ov::op::v6::ExperimentalDetectronPriorGridGenerator::Attributes> attributes = {
        {true, 0, 0, 4.0f, 4.0f},
        {false, 0, 0, 8.0f, 8.0f},
};

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalDetectronPriorGridGenerator, ExperimentalDetectronPriorGridGeneratorLayerTest,
     ::testing::Combine(
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(attributes),
        ::testing::Values(ov::element::f32),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
     ExperimentalDetectronPriorGridGeneratorLayerTest::getTestCaseName);

} // namespace
