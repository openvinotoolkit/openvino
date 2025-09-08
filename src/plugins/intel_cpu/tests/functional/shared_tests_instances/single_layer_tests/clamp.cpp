// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/clamp.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ClampLayerTest;

const std::vector<std::vector<ov::Shape>> input_shapes_static = {
    {{ 50 }},
    {{ 10, 10 }},
    {{ 1, 20, 20 }}
};


const std::vector<std::pair<float, float>> intervals = {
    {-20.1, -10.5},
    {-10.0, 10.0},
    {10.3, 20.4}
};

const std::vector<std::pair<float, float>> intervals_unsigned = {
    {0.1, 10.1},
    {10.0, 100.0},
    {10.6, 20.6}
};

const std::vector<ov::element::Type> model_type = {
    ov::element::f32,
    ov::element::f16,
    ov::element::i64,
    ov::element::i32
};

const auto test_Clamp_signed = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
    ::testing::ValuesIn(intervals),
    ::testing::ValuesIn(model_type),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

const auto test_Clamp_unsigned = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
    ::testing::ValuesIn(intervals_unsigned),
    ::testing::Values(ov::element::u64),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsClamp_signed, ClampLayerTest, test_Clamp_signed, ClampLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsClamp_unsigned, ClampLayerTest, test_Clamp_unsigned, ClampLayerTest::getTestCaseName);
} // namespace
