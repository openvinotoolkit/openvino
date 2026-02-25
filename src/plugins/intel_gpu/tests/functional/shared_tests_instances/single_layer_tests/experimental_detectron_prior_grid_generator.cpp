// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/experimental_detectron_prior_grid_generator.hpp"

namespace {
using ov::test::ExperimentalDetectronPriorGridGeneratorLayerTest;

std::vector<std::vector<ov::test::InputShape>> shapes = {
    ov::test::static_shapes_to_test_representation({{3, 4}, {1, 16, 4, 5}, {1, 3, 100, 200}}),
    ov::test::static_shapes_to_test_representation({{3, 4}, {1, 16, 3, 7}, {1, 3, 100, 200}}),
    // task #72587
    //        ov::test::static_shapes_to_test_representation({{3, 4}, {1, 16, 100, 100}, {1, 3, 100, 200}})
    //        ov::test::static_shapes_to_test_representation({{3, 4}, {1, 16, 100, 100}, {1, 3, 100, 200}})
};

std::vector<ov::op::v6::ExperimentalDetectronPriorGridGenerator::Attributes> attributes = {
        // flatten = true (output tensor is 2D)
        {true, 0, 0, 4.0f, 4.0f},
        // flatten = false (output tensor is 4D)
        {false, 0, 0, 8.0f, 8.0f},
        // task #72587
        //        {true, 3, 6, 64.0f, 64.0f},
        //        {false, 5, 3, 32.0f, 32.0f},
};

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalDetectronPriorGridGenerator_f32,
                         ExperimentalDetectronPriorGridGeneratorLayerTest,
                         testing::Combine(testing::ValuesIn(shapes),
                                          testing::ValuesIn(attributes),
                                          testing::Values(ov::element::f32),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         ExperimentalDetectronPriorGridGeneratorLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ExperimentalDetectronPriorGridGenerator_f16,
                         ExperimentalDetectronPriorGridGeneratorLayerTest,
                         testing::Combine(testing::ValuesIn(shapes),
                                          testing::ValuesIn(attributes),
                                          testing::Values(ov::element::f16),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         ExperimentalDetectronPriorGridGeneratorLayerTest::getTestCaseName);
}  // namespace