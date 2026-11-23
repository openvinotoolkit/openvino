// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/minimum_maximum.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/core.hpp"
#include <cmath>
#include <limits>

namespace {
using ov::test::MaxMinLayerTest;
using ov::test::utils::InputLayerType;
using ov::test::utils::MinMaxOpType;

const std::vector<std::vector<ov::Shape>> input_shapes_static = {
        {{2}, {1}},
        {{1, 1, 1, 3}, {1}},
        {{1, 2, 4}, {1}},
        {{1, 4, 4}, {1}},
        {{1, 4, 4, 1}, {1}},
        {{256, 56}, {256, 56}},
        {{8, 1, 6, 1}, {7, 1, 5}},
};

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16,
};

const std::vector<MinMaxOpType> op_types = {
        MinMaxOpType::MINIMUM,
        MinMaxOpType::MAXIMUM,
};

const std::vector<InputLayerType> second_input_types = {
        InputLayerType::CONSTANT,
        InputLayerType::PARAMETER,
};

INSTANTIATE_TEST_SUITE_P(smoke_maximum, MaxMinLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                                ::testing::ValuesIn(op_types),
                                ::testing::ValuesIn(model_types),
                                ::testing::ValuesIn(second_input_types),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        MaxMinLayerTest::getTestCaseName);

TEST(smoke_MaxMin, FloatNaNIsPreservedAndCommutative) {
    const auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});
    const auto b = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});

    const auto max_op = std::make_shared<ov::op::v1::Maximum>(a, b);
    const auto min_op = std::make_shared<ov::op::v1::Minimum>(a, b);
    const auto model_max = std::make_shared<ov::Model>(max_op, ov::ParameterVector{a, b});
    const auto model_min = std::make_shared<ov::Model>(min_op, ov::ParameterVector{a, b});

    ov::Core core;
    auto compiled_max = core.compile_model(model_max, ov::test::utils::DEVICE_CPU);
    auto compiled_min = core.compile_model(model_min, ov::test::utils::DEVICE_CPU);

    auto req_max = compiled_max.create_infer_request();
    auto req_min = compiled_min.create_infer_request();

    ov::Tensor tensor_nan(ov::element::f32, ov::Shape{4});
    tensor_nan.data<float>()[0] = std::numeric_limits<float>::quiet_NaN();
    tensor_nan.data<float>()[1] = -1.0f;
    tensor_nan.data<float>()[2] = 0.5f;
    tensor_nan.data<float>()[3] = 2.0f;

    ov::Tensor tensor_zero(ov::element::f32, ov::Shape{4});
    std::fill_n(tensor_zero.data<float>(), 4, 0.0f);

    // Maximum(nan, 0)
    req_max.set_input_tensor(0, tensor_nan);
    req_max.set_input_tensor(1, tensor_zero);
    req_max.infer();
    EXPECT_TRUE(std::isnan(req_max.get_output_tensor().data<const float>()[0]));
    EXPECT_FLOAT_EQ(req_max.get_output_tensor().data<const float>()[1], 0.0f);
    EXPECT_FLOAT_EQ(req_max.get_output_tensor().data<const float>()[2], 0.5f);
    EXPECT_FLOAT_EQ(req_max.get_output_tensor().data<const float>()[3], 2.0f);

    // Maximum(0, nan) - commutativity
    req_max.set_input_tensor(0, tensor_zero);
    req_max.set_input_tensor(1, tensor_nan);
    req_max.infer();
    EXPECT_TRUE(std::isnan(req_max.get_output_tensor().data<const float>()[0]));
    EXPECT_FLOAT_EQ(req_max.get_output_tensor().data<const float>()[1], 0.0f);
    EXPECT_FLOAT_EQ(req_max.get_output_tensor().data<const float>()[2], 0.5f);
    EXPECT_FLOAT_EQ(req_max.get_output_tensor().data<const float>()[3], 2.0f);

    // Minimum(nan, 0)
    req_min.set_input_tensor(0, tensor_nan);
    req_min.set_input_tensor(1, tensor_zero);
    req_min.infer();
    EXPECT_TRUE(std::isnan(req_min.get_output_tensor().data<const float>()[0]));
    EXPECT_FLOAT_EQ(req_min.get_output_tensor().data<const float>()[1], -1.0f);
    EXPECT_FLOAT_EQ(req_min.get_output_tensor().data<const float>()[2], 0.0f);
    EXPECT_FLOAT_EQ(req_min.get_output_tensor().data<const float>()[3], 0.0f);

    // Minimum(0, nan) - commutativity
    req_min.set_input_tensor(0, tensor_zero);
    req_min.set_input_tensor(1, tensor_nan);
    req_min.infer();
    EXPECT_TRUE(std::isnan(req_min.get_output_tensor().data<const float>()[0]));
    EXPECT_FLOAT_EQ(req_min.get_output_tensor().data<const float>()[1], -1.0f);
    EXPECT_FLOAT_EQ(req_min.get_output_tensor().data<const float>()[2], 0.0f);
    EXPECT_FLOAT_EQ(req_min.get_output_tensor().data<const float>()[3], 0.0f);
}

}  // namespace
