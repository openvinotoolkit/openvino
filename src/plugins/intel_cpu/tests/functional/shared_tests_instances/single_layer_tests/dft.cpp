// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/dft.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::DFTLayerTest;

const std::vector<ov::test::utils::DFTOpType> op_types = {
    ov::test::utils::DFTOpType::FORWARD,
    ov::test::utils::DFTOpType::INVERSE
};

const std::vector<ov::element::Type> input_type = {
    ov::element::f32,
    ov::element::bf16
};

const std::vector<std::vector<ov::Shape>> input_shapes_static = {
    {{10, 4, 20, 32, 2}},
    {{2, 5, 7, 8, 2}},
    {{1, 120, 128, 1, 2}},
};

/* 1D DFT */

const std::vector<std::vector<int64_t>> axes1D = {
    {0}, {1}, {2}, {3}, {-2}
};

const std::vector<std::vector<int64_t>> signalSizes1D = {
    {}, {16}, {40}
};

const auto testCase1D = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
    ::testing::ValuesIn(input_type),
    ::testing::ValuesIn(axes1D),
    ::testing::ValuesIn(signalSizes1D),
    ::testing::ValuesIn(op_types),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

/* 2D DFT */

const std::vector<std::vector<int64_t>> axes2D = {
    {0, 1}, {2, 1}, {2, 3}, {2, 0}, {1, 3}, {-1, -2}
};
const std::vector<std::vector<int64_t>> signalSizes2D = {
    {}, {5, 7}, {4, 10}, {16, 8}
};

const auto testCase2D = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
    ::testing::ValuesIn(input_type),
    ::testing::ValuesIn(axes2D),
    ::testing::ValuesIn(signalSizes2D),
    ::testing::ValuesIn(op_types),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

/* 3D DFT */

const std::vector<std::vector<int64_t>> axes3D = {
    {0, 1, 2}, {1, 2, 3}, {0, 2, 3}, {2, 3, 1}, {-3, -1, -2},
};

const std::vector<std::vector<int64_t>> signalSizes3D = {
    {}, {4, 8, 16}, {7, 11, 32}
};

const auto testCase3D = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
    ::testing::ValuesIn(input_type),
    ::testing::ValuesIn(axes3D),
    ::testing::ValuesIn(signalSizes3D),
    ::testing::ValuesIn(op_types),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

/* 4D DFT */

const std::vector<std::vector<int64_t>> axes4D = {
    {0, 1, 2, 3}, {-1, 2, 0, 1}
};

const std::vector<std::vector<int64_t>> signalSizes4D = {
    {}, {5, 2, 5, 2}
};

const auto testCase4D = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
    ::testing::ValuesIn(input_type),
    ::testing::ValuesIn(axes4D),
    ::testing::ValuesIn(signalSizes4D),
    ::testing::ValuesIn(op_types),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsDFT_1d, DFTLayerTest, testCase1D, DFTLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsDFT_2d, DFTLayerTest, testCase2D, DFTLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsDFT_3d, DFTLayerTest, testCase3D, DFTLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsDFT_4d, DFTLayerTest, testCase4D, DFTLayerTest::getTestCaseName);
} // namespace
