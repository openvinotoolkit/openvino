// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/mat_mul.hpp"

namespace {
using ov::test::MatMulLayerTest;
using ov::test::utils::InputLayerType;

const std::vector<ov::element::Type> inputPrecisions = {
    ov::element::f32,
    ov::element::f16,
};

std::vector<std::vector<ov::Shape>> no_transpose_shapeRelatedParams = {
        { {2, 1, 1, 5, 6}, {1, 1, 6, 4} },
        { {2, 1, 2, 3, 5, 6}, {1, 1, 6, 4} },
        { {1, 4, 5, 6}, {1, 4, 6, 4} },
        { {4, 5, 6}, {6, 3} },
        { {9, 9, 9}, {9, 9} },
        { {1, 2, 3}, {1, 1, 3, 2} },
        { {1, 3, 2, 4}, {2, 1, 4, 2} },
        { {2, 1, 2, 4}, {1, 3, 4, 2} },
        { {3, 2, 4}, {2, 1, 4, 2} },
        { {2, 1, 4, 2}, {3, 2, 4} },
        { {3}, {2, 2, 3, 1} },
        { {2, 2, 1, 3}, {3} },
        { {1, 5}, {5, 1} },
        { {1, 5}, {5} },
        { {5}, {5, 1} },
        { {5}, {5} },
};

std::vector<std::vector<ov::Shape>> first_transpose_shapeRelatedParams = {
        { {2, 1, 2, 3}, {3, 2, 4} },
        { {100, 65}, {100, 73} },
        { {5, 1}, {5, 1} },
};

std::vector<std::vector<ov::Shape>> second_transpose_shapeRelatedParams = {
        { {1, 16, 128}, {1, 64, 128} },
        { {2, 1, 3, 2}, {3, 4, 2} },
        { {1, 64, 80}, {1, 77, 80} },
        { {65, 100}, {73, 100} },
        { {1, 5}, {1, 5} },
};

std::vector<std::vector<ov::Shape>> both_transpose_shapeRelatedParams = {
        { {2, 2, 4, 16}, {1, 1, 1, 4} },
        { {2, 1, 2, 3}, {3, 4, 2} },
        { {100, 65}, {73, 100} },
        { {5}, {5} },
};

std::vector<InputLayerType> secondaryInputTypes = {
        InputLayerType::CONSTANT,
        InputLayerType::PARAMETER,
};

std::map<std::string, std::string> additional_config = {};

INSTANTIATE_TEST_SUITE_P(smoke_MatMul_NoTranspose, MatMulLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(no_transpose_shapeRelatedParams)),
                ::testing::Values(std::make_pair(false, false)),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::Values(ov::test::utils::DEVICE_GPU),
                ::testing::Values(additional_config)),
        MatMulLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMul_FirstTranspose, MatMulLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(first_transpose_shapeRelatedParams)),
                ::testing::Values(std::make_pair(true, false)),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::Values(ov::test::utils::DEVICE_GPU),
                ::testing::Values(additional_config)),
        MatMulLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMul_SecondTranspose, MatMulLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(second_transpose_shapeRelatedParams)),
                ::testing::Values(std::make_pair(false, true)),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::Values(ov::test::utils::DEVICE_GPU),
                ::testing::Values(additional_config)),
        MatMulLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MatMul_BothTranspose, MatMulLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(both_transpose_shapeRelatedParams)),
                ::testing::Values(std::make_pair(true, true)),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::Values(ov::test::utils::DEVICE_GPU),
                ::testing::Values(additional_config)),
        MatMulLayerTest::getTestCaseName);

} // namespace
