// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <limits>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "single_op_tests/clamp.hpp"

namespace {
using ov::test::ClampLayerTest;

const std::vector<std::vector<ov::Shape>> input_shapes_static = {
    {{ 50 }},
    {{ 10, 10 }},
    {{ 1, 20, 20 }}
};


const std::vector<std::pair<double, double>> intervals = {
    {-20.1, -10.5},
    {-10.0, 10.0},
    {10.3, 20.4}
};

const std::vector<std::pair<double, double>> intervals_unsigned = {
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

const auto test_Clamp_i64 = ::testing::Combine(
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
    ::testing::Values(std::pair<double, double>({static_cast<double>(std::numeric_limits<int64_t>::min()),
                                                 static_cast<double>(std::numeric_limits<int64_t>::max())})),
    ::testing::Values(ov::element::i64),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsClamp_signed, ClampLayerTest, test_Clamp_signed, ClampLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsClamp_unsigned, ClampLayerTest, test_Clamp_unsigned, ClampLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsClamp_i64, ClampLayerTest, test_Clamp_i64, ClampLayerTest::getTestCaseName);

TEST(smoke_Clamp, Int32MaxMinusOneBoundIsPreserved) {
    const auto maximum = std::numeric_limits<int32_t>::max();
    const auto bound = maximum - 1;
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
    const auto clamp = std::make_shared<ov::op::v0::Clamp>(input, 0.0, static_cast<double>(bound));
    const auto model = std::make_shared<ov::Model>(clamp, ov::ParameterVector{input});

    ov::Core core;
    auto compiledModel = core.compile_model(model, ov::test::utils::DEVICE_CPU);
    auto inferRequest = compiledModel.create_infer_request();
    ov::Tensor inputTensor(ov::element::i32, {2});
    inputTensor.data<int32_t>()[0] = bound;
    inputTensor.data<int32_t>()[1] = maximum;
    inferRequest.set_input_tensor(inputTensor);
    inferRequest.infer();

    const auto output = inferRequest.get_output_tensor();
    ASSERT_EQ(output.data<const int32_t>()[0], bound);
    ASSERT_EQ(output.data<const int32_t>()[1], bound);
}
} // namespace
