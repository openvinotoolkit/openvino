// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/transpose.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::TransposeLayerTest;

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
};

/**
 * 4D permute tests
 */
const std::vector<std::vector<ov::Shape>> inputShapes = {
        {{1, 3, 100, 100}},
        // use permute_8x8_4x4 kernel
        {{2, 8, 64, 64}},
        {{2, 5, 64, 64}},
        {{2, 8, 64, 5}},
        {{2, 5, 64, 5}},
};

const std::vector<std::vector<size_t>> inputOrder = {
        // use permute_ref
        std::vector<size_t>{0, 3, 2, 1},
        std::vector<size_t>{},
        // use permute_8x8_4x4 kernel
        std::vector<size_t>{0, 2, 3, 1},
};

INSTANTIATE_TEST_SUITE_P(smoke_Transpose,
                         TransposeLayerTest,
                         testing::Combine(testing::ValuesIn(inputOrder),
                                          testing::ValuesIn(netPrecisions),
                                          testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes)),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         TransposeLayerTest::getTestCaseName);

/**
 * 5D permute tests
 */
const std::vector<std::vector<ov::Shape>> inputShapes5D = {
        {{2, 3, 4, 12, 64}},
        {{2, 5, 11, 32, 32}},
        {{2, 8, 64, 32, 5}},
        {{2, 5, 64, 32, 5}},
};

const std::vector<std::vector<size_t>> inputOrder5D = {
        // use permute_ref
        std::vector<size_t>{0, 3, 4, 2, 1},
        std::vector<size_t>{},
        // use permute_8x8_4x4 kernel
        std::vector<size_t>{0, 2, 3, 4, 1},
        // use permute_kernel_bfzyx_bfyxz
        std::vector<size_t>{0, 1, 3, 4, 2},
};

INSTANTIATE_TEST_SUITE_P(smoke_Transpose_5D,
                         TransposeLayerTest,
                         testing::Combine(testing::ValuesIn(inputOrder5D),
                                          testing::ValuesIn(netPrecisions),
                                          testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes5D)),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         TransposeLayerTest::getTestCaseName);

/**
 * 6D permute tests
 */
const std::vector<std::vector<ov::Shape>> inputShapes6D = {
        {{2, 8, 5, 13, 11, 16}},
        {{2, 11, 6, 2, 15, 10}},
        {{2, 13, 1, 3, 14, 32}},
        {{2, 14, 3, 4, 4, 22}},
};

const std::vector<std::vector<size_t>> inputOrder6D = {
        // use permute_ref
        std::vector<size_t>{0, 4, 3, 5, 2, 1},
        std::vector<size_t>{},
        // use permute_8x8_4x4 kernel
        std::vector<size_t>{0, 2, 3, 4, 5, 1},
};

INSTANTIATE_TEST_SUITE_P(smoke_Transpose_6D,
                         TransposeLayerTest,
                         testing::Combine(testing::ValuesIn(inputOrder6D),
                                          testing::ValuesIn(netPrecisions),
                                          testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes6D)),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         TransposeLayerTest::getTestCaseName);

/**
 * 8D permute tests
 */
const std::vector<std::vector<ov::Shape>> inputShapes8D = {
        {{1, 2, 3, 4, 5, 6, 7, 8}},
};

const std::vector<std::vector<size_t>> inputOrder8D = {
        std::vector<size_t>{1, 2, 4, 3, 6, 7, 5, 0},
};

INSTANTIATE_TEST_SUITE_P(smoke_Transpose_8D,
                         TransposeLayerTest,
                         testing::Combine(testing::ValuesIn(inputOrder8D),
                                          testing::ValuesIn(netPrecisions),
                                          testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes8D)),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         TransposeLayerTest::getTestCaseName);

}  // namespace
