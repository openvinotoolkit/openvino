// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/cum_sum.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::CumSumLayerTest;

const std::vector<std::vector<ov::Shape>> shapes_static = {
    {{16}},
    {{9, 15}},
    {{16, 10, 12}},
    {{5, 14, 5, 7}},
    {{7, 8, 6, 7, 13}},
    {{2, 3, 4, 2, 3, 5}},
    {{4, 3, 6, 2, 3, 4, 5, 2, 3, 4}},
};

const std::vector<ov::element::Type> model_types = {
    ov::element::i8,
    ov::element::u8,
    ov::element::i16,
    ov::element::i32,
    ov::element::f32
};

const std::vector<int64_t> axes = { 0, 1, 2, 3, 4, 5, 6};
const std::vector<int64_t> negativeAxes = { -1, -2, -3, -4, -5, -6 };

const std::vector<bool> exclusive = {true, false};
const std::vector<bool> reverse =   {true, false};

const auto testCasesNegativeAxis = ::testing::Combine(
    ::testing::Values(ov::test::static_shapes_to_test_representation({{4, 16, 3, 6, 5, 2}})),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(negativeAxes),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto testCasesAxis_0 = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes_static)),
    ::testing::ValuesIn(model_types),
    ::testing::Values(axes[0]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto testCasesAxis_1 = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
        std::vector<std::vector<ov::Shape>>(shapes_static.begin() + 1, shapes_static.end()))),
    ::testing::ValuesIn(model_types),
    ::testing::Values(axes[1]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto testCasesAxis_2 = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
        std::vector<std::vector<ov::Shape>>(shapes_static.begin() + 2, shapes_static.end()))),
    ::testing::ValuesIn(model_types),
    ::testing::Values(axes[2]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto testCasesAxis_3 = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
        std::vector<std::vector<ov::Shape>>(shapes_static.begin() + 3, shapes_static.end()))),
    ::testing::ValuesIn(model_types),
    ::testing::Values(axes[3]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto testCasesAxis_4 = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
        std::vector<std::vector<ov::Shape>>(shapes_static.begin() + 4, shapes_static.end()))),
    ::testing::ValuesIn(model_types),
    ::testing::Values(axes[4]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto testCasesAxis_5 = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
        std::vector<std::vector<ov::Shape>>(shapes_static.begin() + 5, shapes_static.end()))),
    ::testing::ValuesIn(model_types),
    ::testing::Values(axes[5]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto testCasesAxis_6 = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
        std::vector<std::vector<ov::Shape>>(shapes_static.begin() + 6, shapes_static.end()))),
    ::testing::ValuesIn(model_types),
    ::testing::Values(axes[6]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsCumSum_negative_axis, CumSumLayerTest, testCasesNegativeAxis, CumSumLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsCumSum_axis_0, CumSumLayerTest, testCasesAxis_0, CumSumLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsCumSum_axis_1, CumSumLayerTest, testCasesAxis_1, CumSumLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsCumSum_axis_2, CumSumLayerTest, testCasesAxis_2, CumSumLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsCumSum_axis_3, CumSumLayerTest, testCasesAxis_3, CumSumLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsCumSum_axis_4, CumSumLayerTest, testCasesAxis_4, CumSumLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsCumSum_axis_5, CumSumLayerTest, testCasesAxis_5, CumSumLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsCumSum_axis_6, CumSumLayerTest, testCasesAxis_6, CumSumLayerTest::getTestCaseName);
} // namespace