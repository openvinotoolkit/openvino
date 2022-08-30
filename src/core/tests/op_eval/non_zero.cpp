// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/non_zero.hpp"

#include <string>
#include <vector>

#include "engines_util/execute_tools.hpp"
#include "gtest/gtest.h"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

OPENVINO_SUPPRESS_DEPRECATED_START

TEST(op_eval, non_zero_0D) {
    auto p = make_shared<op::Parameter>(element::i32, Shape{});
    auto non_zero = make_shared<op::v3::NonZero>(p, element::i64);
    auto fun = make_shared<Function>(OutputVector{non_zero}, ParameterVector{p});

    std::vector<std::vector<int32_t>> inputs{{-1}, {1}, {20}};
    std::vector<std::vector<int64_t>> expected_result{{0}, {0}, {0}};

    for (size_t i = 0; i < inputs.size(); i++) {
        auto result = make_shared<HostTensor>();
        ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::i32>(Shape{}, inputs[i])}));
        EXPECT_EQ(result->get_element_type(), element::i64);
        EXPECT_EQ(result->get_shape(), (Shape{1, 1}));
        auto result_data = read_vector<int64_t>(result);
        ASSERT_EQ(result_data, expected_result[i]);
    }
}

TEST(op_eval, non_zero_0D_0) {
    auto p = make_shared<op::Parameter>(element::i32, Shape{});
    auto non_zero = make_shared<op::v3::NonZero>(p, element::i64);
    auto fun = make_shared<Function>(OutputVector{non_zero}, ParameterVector{p});

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::i32>(Shape{}, {0})}));
    EXPECT_EQ(result->get_element_type(), element::i64);
    EXPECT_EQ(result->get_shape(), (Shape{0, 0}));
    auto result_data = read_vector<int64_t>(result);
    ASSERT_EQ(result_data.data(), nullptr);
}

TEST(op_eval, non_zero_1D) {
    Shape p_shape{5};
    auto p = make_shared<op::Parameter>(element::f32, p_shape);
    auto non_zero = make_shared<op::v3::NonZero>(p, element::i32);
    auto fun = make_shared<Function>(OutputVector{non_zero}, ParameterVector{p});
    std::vector<std::vector<float>> inputs{{1.0, 0, 3.0, 4.0, 0}, {0, 0, 0, 1.0, 3.2}, {1.0, 1.0, 1.0, 1.0, 1.0}};
    std::vector<std::vector<int32_t>> expected_result{{0, 2, 3}, {3, 4}, {0, 1, 2, 3, 4}};
    std::vector<Shape> expected_output_shape{Shape{1, 3}, Shape{1, 2}, Shape{1, 5}};
    for (size_t i = 0; i < inputs.size(); i++) {
        auto result = make_shared<HostTensor>();
        ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(p_shape, inputs[i])}));
        EXPECT_EQ(result->get_element_type(), element::i32);
        EXPECT_EQ(result->get_shape(), expected_output_shape[i]);
        auto result_data = read_vector<int32_t>(result);
        ASSERT_EQ(result_data, expected_result[i]);
    }
}

TEST(op_eval, non_zero_1D_0s) {
    Shape p_shape{5};
    auto p = make_shared<op::Parameter>(element::f32, p_shape);
    auto non_zero = make_shared<op::v3::NonZero>(p, element::i64);
    auto fun = make_shared<Function>(OutputVector{non_zero}, ParameterVector{p});

    std::vector<float> input(shape_size(p_shape), 0);
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(p_shape, input)}));
    EXPECT_EQ(result->get_element_type(), element::i64);
    EXPECT_EQ(result->get_shape(), (Shape{1, 0}));
    auto result_data = read_vector<int64_t>(result);
    ASSERT_EQ(result_data.data(), nullptr);
}

TEST(op_eval, non_zero_2D) {
    Shape p_shape{3, 2};
    auto p = make_shared<op::Parameter>(element::i32, p_shape);
    auto non_zero = make_shared<op::v3::NonZero>(p);
    auto fun = make_shared<Function>(OutputVector{non_zero}, ParameterVector{p});
    std::vector<std::vector<int32_t>> inputs{{1, 0, 3, 4, 0, 1}, {0, 0, 0, 0, 1, 3}, {1, 1, 1, 1, 1, 1}};
    std::vector<std::vector<int64_t>> expected_result{{0, 1, 1, 2, 0, 0, 1, 1},
                                                      {2, 2, 0, 1},
                                                      {0, 0, 1, 1, 2, 2, 0, 1, 0, 1, 0, 1}};
    std::vector<Shape> expected_output_shape{Shape{2, 4}, Shape{2, 2}, Shape{2, 6}};
    for (size_t i = 0; i < inputs.size(); i++) {
        auto result = make_shared<HostTensor>();
        ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::i32>(p_shape, inputs[i])}));
        EXPECT_EQ(result->get_element_type(), element::i64);
        EXPECT_EQ(result->get_shape(), expected_output_shape[i]);
        auto result_data = read_vector<int64_t>(result);
        ASSERT_EQ(result_data, expected_result[i]);
    }
}

TEST(op_eval, non_zero_3D) {
    Shape p_shape{3, 2, 2};
    auto p = make_shared<op::Parameter>(element::i64, p_shape);
    auto non_zero = make_shared<op::v3::NonZero>(p, element::i32);
    auto fun = make_shared<Function>(OutputVector{non_zero}, ParameterVector{p});
    std::vector<std::vector<int64_t>> inputs{{1, 0, 3, 4, 0, 1, 0, 0, 1, 3, 5, 0},
                                             {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
    std::vector<std::vector<int32_t>> expected_result{
        {0, 0, 0, 1, 2, 2, 2, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0},
        {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}};
    std::vector<Shape> expected_output_shape{Shape{3, 7}, Shape{3, 12}};
    for (size_t i = 0; i < inputs.size(); i++) {
        auto result = make_shared<HostTensor>();
        ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::i64>(p_shape, inputs[i])}));
        EXPECT_EQ(result->get_element_type(), element::i32);
        EXPECT_EQ(result->get_shape(), expected_output_shape[i]);
        auto result_data = read_vector<int32_t>(result);
        ASSERT_EQ(result_data, expected_result[i]);
    }
}

TEST(op_eval, non_zero_3D_0s) {
    Shape p_shape{3, 2, 2};
    auto p = make_shared<op::Parameter>(element::i64, p_shape);
    auto non_zero = make_shared<op::v3::NonZero>(p, element::i32);
    auto fun = make_shared<Function>(OutputVector{non_zero}, ParameterVector{p});

    std::vector<int64_t> input(shape_size(p_shape), 0);
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::i64>(p_shape, input)}));
    EXPECT_EQ(result->get_element_type(), element::i32);
    EXPECT_EQ(result->get_shape(), (Shape{p_shape.size(), 0}));
    auto result_data = read_vector<int32_t>(result);
    ASSERT_EQ(result_data.data(), nullptr);
}

TEST(op_eval, non_zero_dynamic) {
    PartialShape p_shape = PartialShape::dynamic();
    auto p = make_shared<op::Parameter>(element::i32, p_shape);
    auto non_zero = make_shared<op::v3::NonZero>(p);
    auto fun = make_shared<Function>(OutputVector{non_zero}, ParameterVector{p});
    std::vector<std::vector<int32_t>> inputs{{1, 0, 3, 4, 0}, {0, 0, 0, 0, 1, 3}, {0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0}};
    std::vector<Shape> input_shapes{Shape{5}, Shape{3, 2}, Shape{3, 2, 2}};
    std::vector<std::vector<int64_t>> expected_result{{0, 2, 3}, {2, 2, 0, 1}, {0, 1, 1, 1, 0, 0}};
    std::vector<Shape> expected_output_shape{Shape{1, 3}, Shape{2, 2}, Shape{3, 2}};
    for (size_t i = 0; i < inputs.size(); i++) {
        auto result = make_shared<HostTensor>();
        ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::i32>(input_shapes[i], inputs[i])}));
        EXPECT_EQ(result->get_element_type(), element::i64);
        EXPECT_EQ(result->get_shape(), expected_output_shape[i]);
        auto result_data = read_vector<int64_t>(result);
        ASSERT_EQ(result_data, expected_result[i]);
    }
}
