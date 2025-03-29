// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/cum_sum.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"

using namespace ov;

TEST(type_prop, cum_sum_op_default_attributes_no_axis_input) {
    PartialShape data_shape{2, 4};
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto cum_sum = std::make_shared<op::v0::CumSum>(A);

    EXPECT_EQ(cum_sum->is_exclusive(), false);
    EXPECT_EQ(cum_sum->is_reverse(), false);
    EXPECT_EQ(cum_sum->get_input_size(), 2);
    EXPECT_EQ(cum_sum->get_element_type(), element::f32);
    EXPECT_EQ(cum_sum->get_output_partial_shape(0), data_shape);
}

TEST(type_prop, cum_sum_op_default_attributes_with_axis_param) {
    PartialShape data_shape{2, 4};
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto axis = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{});
    auto cum_sum = std::make_shared<op::v0::CumSum>(A, axis);

    EXPECT_EQ(cum_sum->is_exclusive(), false);
    EXPECT_EQ(cum_sum->is_reverse(), false);
    EXPECT_EQ(cum_sum->get_input_size(), 2);
    EXPECT_EQ(cum_sum->get_element_type(), element::f32);
    EXPECT_EQ(cum_sum->get_output_partial_shape(0), data_shape);
}

TEST(type_prop, cum_sum_op_default_attributes_with_axis_const) {
    PartialShape data_shape{2, 4};
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto axis = ov::op::v0::Constant::create(element::i32, Shape{}, {1});
    auto cum_sum = std::make_shared<op::v0::CumSum>(A, axis);

    EXPECT_EQ(cum_sum->is_exclusive(), false);
    EXPECT_EQ(cum_sum->is_reverse(), false);
    EXPECT_EQ(cum_sum->get_input_size(), 2);
    EXPECT_EQ(cum_sum->get_element_type(), element::f32);
    EXPECT_EQ(cum_sum->get_output_partial_shape(0), data_shape);
}

TEST(type_prop, cum_sum_op_custom_attributes) {
    PartialShape data_shape{2, 4};
    auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
    auto axis = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{});
    bool exclusive = true;
    bool reverse = true;
    auto cum_sum = std::make_shared<op::v0::CumSum>(A, axis, exclusive, reverse);

    EXPECT_EQ(cum_sum->is_exclusive(), true);
    EXPECT_EQ(cum_sum->is_reverse(), true);
    EXPECT_EQ(cum_sum->get_input_size(), 2);
    EXPECT_EQ(cum_sum->get_element_type(), element::f32);
    EXPECT_EQ(cum_sum->get_output_partial_shape(0), data_shape);
}

TEST(type_prop, cum_sum_op_data_shapes) {
    std::vector<PartialShape> input_shpes{{},
                                          {10},
                                          {3, 5},
                                          {2, 3, 5},
                                          {4, 16, 64, 32},
                                          {2, 4, 16, 64, 32},
                                          {2, Dimension(2, 4), Dimension()},
                                          PartialShape::dynamic()};

    for (auto& shape : input_shpes) {
        try {
            auto axis = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{});
            auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, shape);
            auto cum_sum = std::make_shared<op::v0::CumSum>(A, axis);

            EXPECT_EQ(cum_sum->get_output_partial_shape(0), shape);
        } catch (...) {
            FAIL() << "Data input shape validation check failed for unexpected reason";
        }
    }
}

TEST(type_prop, cum_sum_op_incorrect_axis_shapes) {
    PartialShape data_shape{2, 4};
    std::vector<PartialShape> incorrect_axis_shpes{{1}, {1, 1}, {2}};
    for (auto& shape : incorrect_axis_shpes) {
        try {
            auto axis = std::make_shared<ov::op::v0::Parameter>(element::i32, shape);
            auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
            auto cum_sum = std::make_shared<op::v0::CumSum>(A, axis);
        } catch (...) {
            FAIL() << "CumSum axis input shape validation shouldn't throw for backward compatibility";
        }
    }
}

TEST(type_prop, cum_sum_op_element_types) {
    PartialShape data_shape{2, 4};
    std::vector<element::Type> element_types{element::u4,
                                             element::u8,
                                             element::u16,
                                             element::u32,
                                             element::i8,
                                             element::i16,
                                             element::i32,
                                             element::i64,
                                             element::f32,
                                             element::f64,
                                             element::u32,
                                             element::boolean};

    for (auto& et : element_types) {
        try {
            auto axis = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{});
            auto A = std::make_shared<ov::op::v0::Parameter>(et, data_shape);

            EXPECT_NO_THROW(const auto unused = std::make_shared<op::v0::CumSum>(A, axis));
        } catch (...) {
            FAIL() << "Data input element type validation check failed for unexpected reason";
        }
    }
}

TEST(type_prop, cum_sum_op_incorrect_axis_element_type) {
    std::vector<element::Type> element_types{element::u32, element::f32, element::boolean, element::u32};

    PartialShape data_shape{2, 4};

    for (auto& et : element_types) {
        try {
            auto axis = std::make_shared<ov::op::v0::Parameter>(et, PartialShape{});
            auto A = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape);
            auto cum_sum = std::make_shared<op::v0::CumSum>(A, axis);

            FAIL() << "Invalid element type of axis input not detected";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "axis element type must be either int64_t or int32_t");
        } catch (...) {
            FAIL() << "Axis input element type validation check failed for unexpected reason";
        }
    }
}
