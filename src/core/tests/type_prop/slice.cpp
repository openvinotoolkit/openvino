// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <dimension_tracker.hpp>
#include <numeric>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;

namespace {
template <typename T>
std::shared_ptr<Node> make_slice_op_const_inputs(const std::vector<std::vector<T>>& args,
                                                 PartialShape& data_shape,
                                                 element::Type_t et) {
    const auto& start_val = args[0];
    const auto& stop_val = args[1];
    const auto& step_val = args[2];

    const auto data = std::make_shared<op::v0::Parameter>(et, data_shape);
    const auto start = std::make_shared<op::v0::Constant>(et, Shape{start_val.size()}, start_val);
    const auto stop = std::make_shared<op::v0::Constant>(et, Shape{stop_val.size()}, stop_val);
    const auto step = std::make_shared<op::v0::Constant>(et, Shape{step_val.size()}, step_val);

    if (args.size() > 3) {
        const auto& axes_val = args[3];
        const auto axes = std::make_shared<op::v0::Constant>(et, Shape{axes_val.size()}, axes_val);
        return std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
    }
    return std::make_shared<op::v8::Slice>(data, start, stop, step);
}
}  // namespace

TEST(type_prop, slice_v8_basic_const_inputs) {
    PartialShape data_shape{10, 10, 10, 10, 10, 10, 10, 10};
    PartialShape expected_out_shape{7, 4, 10, 10, 9, 9, 5, 10};

    std::vector<int32_t> start_val{1, 1, -20, 9, 9, 9, 9, 20};
    std::vector<int32_t> stop_val{8, 8, 20, -11, 0, -10, -11, -20};
    std::vector<int32_t> step_val{1, 2, 1, -1, -1, -1, -2, -1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    element::Type_t et = element::i32;

    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_dif_steps) {
    PartialShape data_shape{5, 5, 5, 5, 9, 9};
    PartialShape expected_out_shape{3, 2, 2, 1, 5, 3};

    std::vector<int32_t> start_val{0, 0, 0, 0, 0, 0};
    std::vector<int32_t> stop_val{5, 5, 5, 5, 9, 9};
    std::vector<int32_t> step_val{2, 3, 4, 5, 2, 3};

    element::Type_t et = element::i32;

    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_dif_neg_steps) {
    PartialShape data_shape{5, 5, 5, 5, 9, 9};
    PartialShape expected_out_shape{2, 2, 1, 1, 4, 3};

    std::vector<int32_t> start_val{5, 5, 5, 5, 9, 9};
    std::vector<int32_t> stop_val{0, 0, 0, 0, 0, 0};
    std::vector<int32_t> step_val{-2, -3, -4, -5, -2, -3};

    element::Type_t et = element::i32;

    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_default_axes) {
    PartialShape data_shape{10, 10, 10, 10, 10, 10, 10, 10};
    PartialShape expected_out_shape{7, 4, 10, 10, 9, 9, 5, 10};

    std::vector<int32_t> start_val{1, 1, -20, 9, 9, 9, 9, 20};
    std::vector<int32_t> stop_val{8, 8, 20, -11, 0, -10, -11, -20};
    std::vector<int32_t> step_val{1, 2, 1, -1, -1, -1, -2, -1};

    element::Type_t et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_const_inputs_output_zero_dims) {
    PartialShape data_shape{5, 5, 5, 5, 9, 9};
    PartialShape expected_out_shape{3, 0, 5, 0, 5, 0};

    std::vector<int32_t> start_val{0, 0, 0, 5, 0, 10};
    std::vector<int32_t> stop_val{5, 5, 5, 5, 9, 9};
    std::vector<int32_t> step_val{2, -3, 1, 5, 2, 1};

    element::Type_t et = element::i32;

    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_unordered_axes) {
    PartialShape data_shape{10, 10, 10, 10, 10, 10, 10, 10};
    PartialShape expected_out_shape{4, 9, 7, 10, 10, 9, 5, 10};

    std::vector<int32_t> start_val{1, 1, -20, 9, 9, 9, 9, 20};
    std::vector<int32_t> stop_val{8, 8, 20, -11, 0, -10, -11, -20};
    std::vector<int32_t> step_val{1, 2, 1, -1, -1, -1, -2, -1};

    std::vector<int32_t> axes_val{2, 0, 3, 7, 1, 5, 6, 4};

    element::Type_t et = element::i32;

    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_const_inputs_not_all_axes_unordered) {
    PartialShape data_shape{10, 10, 10, 10, 10, 20, Dimension(20, 30), 30, Dimension(2, 5), Dimension(-1)};
    PartialShape expected_out_shape{4, 7, 10, 10, 9, 20, Dimension(10, 15), 30, Dimension(2, 5), Dimension(-1)};

    std::vector<int32_t> start_val{1, 1, -20, 9, 10, 9};
    std::vector<int32_t> stop_val{8, 8, 20, -11, 25, 0};
    std::vector<int32_t> step_val{1, 2, 1, -1, 1, -1};

    std::vector<int32_t> axes_val{1, 0, 2, 3, 6, 4};

    element::Type_t et = element::i32;

    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_data_dynamic_bounds_dimensions) {
    PartialShape data_shape{Dimension(1, 5), Dimension(10, 20), Dimension(20, 30), 16, Dimension(30, 40)};
    PartialShape expected_out_shape{1, 6, Dimension(10, 15), 16, Dimension(0, 5)};

    std::vector<int32_t> start_val{0, 2, 10, 0, 35};
    std::vector<int32_t> stop_val{1, 8, 25, 16, 40};
    std::vector<int32_t> step_val{1, 1, 1, 1, 1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    element::Type_t et = element::i32;

    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_data_dynamic_rank) {
    PartialShape data_shape = PartialShape::dynamic();
    PartialShape expected_out_shape = PartialShape::dynamic();

    std::vector<int32_t> start_val{1, 1, -20, 9, 9, 9, 9, 20};
    std::vector<int32_t> stop_val{8, 8, 20, -11, 0, -10, -11, -20};
    std::vector<int32_t> step_val{1, 2, 1, -1, -1, -1, -2, -1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    element::Type_t et = element::i32;

    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
    EXPECT_TRUE(op->get_output_partial_shape(0).rank().is_dynamic());
}

TEST(type_prop, slice_v8_basic_param_inputs_default_axes) {
    PartialShape data_shape{Dimension(0, 10),
                            Dimension(1, 10),
                            10,
                            Dimension(3, 5),
                            Dimension(-1, -1),
                            Dimension(100, -1),
                            Dimension(0, 8),
                            Dimension(4, 8),
                            16};
    PartialShape expected_out_shape{Dimension(0, 10),
                                    Dimension(0, 10),
                                    Dimension(0, 10),
                                    Dimension(0, 5),
                                    Dimension(0, -1),
                                    Dimension(0, -1),
                                    Dimension(0, 8),
                                    Dimension(4, 8),
                                    16};

    PartialShape start_shape{7};
    PartialShape stop_shape{7};
    PartialShape step_shape{7};

    element::Type_t et = element::i32;

    const auto data = std::make_shared<op::v0::Parameter>(et, data_shape);
    const auto start = std::make_shared<op::v0::Parameter>(et, start_shape);
    const auto stop = std::make_shared<op::v0::Parameter>(et, stop_shape);
    const auto step = std::make_shared<op::v0::Parameter>(et, step_shape);

    const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_sss_param_inputs_mixed_neg_const_axes) {
    PartialShape data_shape{Dimension(5, 10), 2, 7, Dimension(0, 4)};
    PartialShape expected_out_shape{Dimension(0, 10), Dimension(0, 2), 7, Dimension(0, 4)};

    PartialShape start_shape{3};
    PartialShape stop_shape{3};
    PartialShape step_shape{3};

    element::Type_t et = element::i32;

    Shape axes_shape{3};
    std::vector<int32_t> axes_val{0, 3, 1};

    const auto data = std::make_shared<op::v0::Parameter>(et, data_shape);
    const auto start = std::make_shared<op::v0::Parameter>(et, start_shape);
    const auto stop = std::make_shared<op::v0::Parameter>(et, stop_shape);
    const auto step = std::make_shared<op::v0::Parameter>(et, step_shape);
    const auto axes = std::make_shared<op::v0::Constant>(et, axes_shape, axes_val);

    const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_sss_param_inputs_mixed_const_axes) {
    PartialShape data_shape{Dimension(5, 10), 2, 7, Dimension(0, 4)};
    PartialShape expected_out_shape{Dimension(0, 10), Dimension(0, 2), 7, Dimension(0, 4)};

    PartialShape start_shape{3};
    PartialShape stop_shape{3};
    PartialShape step_shape{3};

    element::Type_t et = element::i32;

    Shape axes_shape{3};
    std::vector<int32_t> axes_val{-4, -1, -3};

    const auto data = std::make_shared<op::v0::Parameter>(et, data_shape);
    const auto start = std::make_shared<op::v0::Parameter>(et, start_shape);
    const auto stop = std::make_shared<op::v0::Parameter>(et, stop_shape);
    const auto step = std::make_shared<op::v0::Parameter>(et, step_shape);
    const auto axes = std::make_shared<op::v0::Constant>(et, axes_shape, axes_val);

    const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_sss_param_inputs_const_axes) {
    PartialShape data_shape{Dimension(0, 10),
                            Dimension(1, 10),
                            10,
                            Dimension(3, 5),
                            Dimension(-1, -1),
                            Dimension(100, -1),
                            Dimension(0, 8),
                            Dimension(4, 8),
                            16};

    PartialShape expected_out_shape{Dimension(0, 10),
                                    Dimension(0, 10),
                                    Dimension(0, 10),
                                    Dimension(0, 5),
                                    Dimension(0, -1),
                                    Dimension(0, -1),
                                    Dimension(0, 8),
                                    Dimension(4, 8),
                                    16};

    PartialShape start_shape{7};
    PartialShape stop_shape{7};
    PartialShape step_shape{7};

    element::Type_t et = element::i32;

    Shape axes_shape{7};
    std::vector<int32_t> axes_val(7);
    std::iota(axes_val.begin(), axes_val.end(), 0);

    const auto data = std::make_shared<op::v0::Parameter>(et, data_shape);
    const auto start = std::make_shared<op::v0::Parameter>(et, start_shape);
    const auto stop = std::make_shared<op::v0::Parameter>(et, stop_shape);
    const auto step = std::make_shared<op::v0::Parameter>(et, step_shape);
    const auto axes = std::make_shared<op::v0::Constant>(et, axes_shape, axes_val);

    const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_sss_param_inputs_param_axes) {
    PartialShape data_shape{Dimension(0, 10),
                            Dimension(1, 10),
                            10,
                            Dimension(3, 5),
                            Dimension(-1, -1),
                            Dimension(100, -1),
                            Dimension(0, 8),
                            Dimension(4, 8),
                            16};

    PartialShape expected_out_shape{Dimension(0, 10),
                                    Dimension(0, 10),
                                    Dimension(0, 10),
                                    Dimension(0, 5),
                                    Dimension(0, -1),
                                    Dimension(0, -1),
                                    Dimension(0, 8),
                                    Dimension(4, 8),
                                    16};
    PartialShape start_shape{7};
    PartialShape stop_shape{7};
    PartialShape step_shape{7};
    PartialShape axes_shape{7};

    element::Type_t et = element::i32;

    const auto data = std::make_shared<op::v0::Parameter>(et, data_shape);
    const auto start = std::make_shared<op::v0::Parameter>(et, start_shape);
    const auto stop = std::make_shared<op::v0::Parameter>(et, stop_shape);
    const auto step = std::make_shared<op::v0::Parameter>(et, step_shape);
    const auto axes = std::make_shared<op::v0::Parameter>(et, axes_shape);

    const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_MAX_MIN_INT_dynamic_dimensions) {
    PartialShape data_shape{10,
                            Dimension(10, 20),
                            Dimension(10, 20),
                            Dimension(10, 20),
                            Dimension(10, 20),
                            Dimension(10, 20),
                            Dimension(30, 40),
                            Dimension(0, 50),
                            Dimension(0, -1)};
    PartialShape expected_out_shape{8,
                                    Dimension(8, 18),
                                    5,
                                    10,
                                    Dimension(10, 15),
                                    Dimension(10, 20),
                                    Dimension(30, 40),
                                    Dimension(0, 50),
                                    Dimension(-1)};

    std::vector<int32_t> start_val{2, 2, INT32_MIN, INT32_MIN, INT32_MIN, INT32_MIN, INT32_MIN, 0, 0};
    std::vector<int32_t> stop_val{10, INT32_MAX, 5, 10, 15, 25, INT32_MAX, 50, INT32_MAX};
    std::vector<int32_t> step_val{1, 1, 1, 1, 1, 1, 1, 1, 1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    element::Type_t et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_MAX_MIN_INT_dynamic_dimensions_neg_step) {
    PartialShape data_shape{10,
                            Dimension(10, 20),
                            Dimension(10, 20),
                            Dimension(10, 20),
                            Dimension(11, 20),
                            Dimension(10, 20),
                            Dimension(10, 20),
                            Dimension(30, 40),
                            Dimension(20),
                            Dimension(20),
                            Dimension(0, 20),
                            Dimension(-1),
                            Dimension(-1)};
    PartialShape expected_out_shape{8,
                                    Dimension(8, 18),
                                    Dimension(5, 15),
                                    Dimension(0, 9),
                                    Dimension(0, 9),
                                    Dimension(0, 4),
                                    Dimension(0),
                                    Dimension(30, 40),
                                    Dimension(20),
                                    Dimension(20),
                                    Dimension(0, 20),
                                    Dimension(0, 21),
                                    Dimension(-1)};

    std::vector<int32_t> start_val{9,
                                   INT32_MAX,
                                   INT32_MAX,
                                   INT32_MAX,
                                   INT32_MAX,
                                   INT32_MAX,
                                   INT32_MAX,
                                   INT32_MAX,
                                   20,
                                   20,
                                   20,
                                   20,
                                   INT32_MAX};
    std::vector<int32_t> stop_val{1, 1, 4, 10, 10, 15, 25, INT32_MIN, -21, INT32_MIN, INT32_MIN, INT32_MIN, INT32_MIN};
    std::vector<int32_t> step_val{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    element::Type_t et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_data_full_dynamic_dims) {
    PartialShape data_shape{Dimension(-1), Dimension(-1), Dimension(-1)};
    PartialShape expected_out_shape{Dimension(0, 6), Dimension(0, 15), Dimension(0, 5)};

    std::vector<int32_t> start_val{2, 10, 35};
    std::vector<int32_t> stop_val{8, 25, 40};
    std::vector<int32_t> step_val{1, 1, 1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    element::Type_t et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_data_full_dynamic_dims_neg_ind) {
    PartialShape data_shape{Dimension(-1), Dimension(-1), Dimension(-1)};
    PartialShape expected_out_shape{Dimension(0, 6), Dimension(0, 15), Dimension(0, 5)};

    std::vector<int32_t> start_val{-8, -25, -40};
    std::vector<int32_t> stop_val{-2, -10, -35};
    std::vector<int32_t> step_val{1, 1, 1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    element::Type_t et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_data_full_dynamic_dims_neg_step) {
    PartialShape data_shape{Dimension(-1), Dimension(-1), Dimension(-1)};
    PartialShape expected_out_shape{Dimension(0, 6), Dimension(0, 15), Dimension(0, 5)};

    std::vector<int32_t> start_val{8, 25, 40};
    std::vector<int32_t> stop_val{2, 10, 35};
    std::vector<int32_t> step_val{-1, -1, -1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    element::Type_t et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_data_full_dynamic_dims_neg_step_neg_ind) {
    PartialShape data_shape{Dimension(-1), Dimension(-1), Dimension(-1)};
    PartialShape expected_out_shape{Dimension(0, 6), Dimension(0, 15), Dimension(0, 5)};

    std::vector<int32_t> start_val{-2, -10, -35};
    std::vector<int32_t> stop_val{-8, -25, -40};
    std::vector<int32_t> step_val{-1, -1, -1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    element::Type_t et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_data_full_dynamic_dims_neg_step_mix_ind) {
    PartialShape data_shape{Dimension(-1), Dimension(-1), Dimension(-1), Dimension(-1), Dimension(-1)};
    PartialShape expected_out_shape{Dimension(0, 6), Dimension(0, 6), Dimension(-1), Dimension(0, -1), Dimension(-1)};

    std::vector<int32_t> start_val{5, 5, -10, INT32_MAX, INT32_MAX};
    std::vector<int32_t> stop_val{-10, INT32_MIN, 5, 5, INT32_MIN};
    std::vector<int32_t> step_val{-1, -1, -1, -1, -1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    element::Type_t et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_dynamic_dims_maxint32) {
    PartialShape data_shape{Dimension(0, 2000), Dimension(-1), 4};
    PartialShape expected_out_shape{Dimension(0, 2000), Dimension(-1), Dimension(4)};

    std::vector<int32_t> start_val{0, 0, 0};
    std::vector<int32_t> stop_val{INT32_MAX, INT32_MAX, INT32_MAX};
    std::vector<int32_t> step_val{1, 1, 1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    element::Type_t et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_dynamic_dims_maxint64) {
    PartialShape data_shape{Dimension(0, 2000), Dimension(-1), 4};
    PartialShape expected_out_shape{Dimension(0, 2000), Dimension(-1), Dimension(4)};

    std::vector<int64_t> start_val{0, 0, 0};
    std::vector<int64_t> stop_val{INT64_MAX, INT64_MAX, INT64_MAX};
    std::vector<int64_t> step_val{1, 1, 1};

    std::vector<int64_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    element::Type_t et = element::i64;
    std::vector<std::vector<int64_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_dynamic_dims_maxint32_start1) {
    PartialShape data_shape{Dimension(0, 2000), Dimension(-1), 4};
    PartialShape expected_out_shape{Dimension(0, 2000), Dimension(-1), Dimension(4)};

    std::vector<int32_t> start_val{1};
    std::vector<int32_t> stop_val{INT32_MAX};
    std::vector<int32_t> step_val{1};

    std::vector<int32_t> axes_val{1};

    element::Type_t et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_dynamic_dims_maxint64_start1) {
    PartialShape data_shape{Dimension(0, 2000), Dimension(-1), 4};
    PartialShape expected_out_shape{Dimension(0, 2000), Dimension(-1), Dimension(4)};

    std::vector<int64_t> start_val{1};
    std::vector<int64_t> stop_val{INT64_MAX};
    std::vector<int64_t> step_val{1};

    std::vector<int64_t> axes_val{1};

    element::Type_t et = element::i64;
    std::vector<std::vector<int64_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_MAX_MIN_INT_64_dynamic_dimensions_neg_step) {
    PartialShape data_shape{10,
                            Dimension(10, 20),
                            Dimension(10, 20),
                            Dimension(10, 20),
                            Dimension(11, 20),
                            Dimension(10, 20),
                            Dimension(10, 20),
                            Dimension(30, 40),
                            Dimension(20),
                            Dimension(20),
                            Dimension(0, 20),
                            Dimension(-1),
                            Dimension(-1)};
    PartialShape expected_out_shape{8,
                                    Dimension(8, 18),
                                    Dimension(5, 15),
                                    Dimension(0, 9),
                                    Dimension(0, 9),
                                    Dimension(0, 4),
                                    Dimension(0),
                                    Dimension(30, 40),
                                    Dimension(20),
                                    Dimension(20),
                                    Dimension(0, 20),
                                    Dimension(0, 21),
                                    Dimension(-1)};

    std::vector<int64_t> start_val{9,
                                   INT64_MAX,
                                   INT64_MAX,
                                   INT64_MAX,
                                   INT64_MAX,
                                   INT64_MAX,
                                   INT64_MAX,
                                   INT64_MAX,
                                   20,
                                   20,
                                   20,
                                   20,
                                   INT64_MAX};
    std::vector<int64_t> stop_val{1, 1, 4, 10, 10, 15, 25, INT64_MIN, -21, INT64_MIN, INT64_MIN, INT64_MIN, INT64_MIN};
    std::vector<int64_t> step_val{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

    std::vector<int64_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    element::Type_t et = element::i64;
    std::vector<std::vector<int64_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_MAX_MIN_64_no_upper_bounds_neg_step) {
    PartialShape data_shape{Dimension(-1), Dimension(0, INT64_MAX), Dimension(0, INT64_MAX)};
    PartialShape expected_out_shape{Dimension(-1), Dimension(0, INT64_MAX), Dimension(0, INT64_MAX)};

    std::vector<int64_t> start_val{INT64_MAX, INT64_MAX, INT64_MAX};
    std::vector<int64_t> stop_val{INT64_MIN, INT64_MIN, INT64_MIN};
    std::vector<int64_t> step_val{-1, -1, -1};

    std::vector<int64_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    element::Type_t et = element::i64;
    std::vector<std::vector<int64_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_MAX_MIN_64_no_upper_bounds) {
    PartialShape data_shape{Dimension(-1), Dimension(0, INT64_MAX), Dimension(0, INT64_MAX)};
    PartialShape expected_out_shape{Dimension(-1), Dimension(0, INT64_MAX), Dimension(0, INT64_MAX)};

    std::vector<int64_t> start_val{INT64_MIN, INT64_MIN, INT64_MIN};
    std::vector<int64_t> stop_val{INT64_MAX, INT64_MAX, INT64_MAX};
    std::vector<int64_t> step_val{1, 1, 1};

    std::vector<int64_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    element::Type_t et = element::i64;
    std::vector<std::vector<int64_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_MAX_MIN_32_no_upper_bounds) {
    PartialShape data_shape{Dimension(-1), Dimension(0, INT64_MAX), Dimension(0, INT32_MAX), Dimension(0, INT32_MAX)};
    PartialShape expected_out_shape{Dimension(-1),
                                    Dimension(0, INT64_MAX),
                                    Dimension(0, INT32_MAX),
                                    Dimension(0, INT32_MAX)};

    std::vector<int32_t> start_val{INT32_MIN, INT32_MIN, INT32_MIN, INT32_MIN};
    std::vector<int32_t> stop_val{INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX};
    std::vector<int32_t> step_val{1, 1, 1, 1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    element::Type_t et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_basic_const_inputs_MAX_MIN_32_no_upper_bounds_neg_step) {
    PartialShape data_shape{Dimension(-1), Dimension(0, INT64_MAX), Dimension(0, INT32_MAX), Dimension(0, INT32_MAX)};
    PartialShape expected_out_shape{Dimension(-1), Dimension(-1), Dimension(0, INT32_MAX), Dimension(0, INT32_MAX)};

    std::vector<int32_t> start_val{INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX};
    std::vector<int32_t> stop_val{INT32_MIN, INT32_MIN, INT32_MIN, INT32_MIN};
    std::vector<int32_t> step_val{-1, -1, -1, -1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    element::Type_t et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_dynamic_dim_zero_start_negative_stop) {
    PartialShape data_shape{Dimension(-1)};
    PartialShape expected_out_shape{Dimension(-1)};

    std::vector<int32_t> start_val{0};
    std::vector<int32_t> stop_val{-2};
    std::vector<int32_t> step_val{1};

    std::vector<int32_t> axes_val{0};

    element::Type_t et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
}

TEST(type_prop, slice_v8_duplicated_axes) {
    PartialShape data_shape{100, 100, 100, 100};
    PartialShape expected_out_shape{100, 100, 100, 100};

    std::vector<int32_t> start_val{2, 10, 35, 10};
    std::vector<int32_t> stop_val{8, 25, 40, 20};
    std::vector<int32_t> step_val{1, 1, 1, 100};

    std::vector<int32_t> axes_val{2, 1, 2, 3};

    element::Type_t et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};

    EXPECT_THROW(make_slice_op_const_inputs(input_vals, data_shape, et), NodeValidationFailure);
}

TEST(type_prop, slice_v8_zero_step) {
    PartialShape data_shape{100, 100, 100};
    PartialShape expected_out_shape{100, 100, 100};

    std::vector<int32_t> start_val{2, 10, 35};
    std::vector<int32_t> stop_val{8, 25, 40};
    std::vector<int32_t> step_val{1, 0, 1};

    std::vector<int32_t> axes_val{1, 2, 3};

    element::Type_t et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};

    EXPECT_THROW(make_slice_op_const_inputs(input_vals, data_shape, et), NodeValidationFailure);
}

TEST(type_prop, slice_v8_ind_bad_type) {
    PartialShape data_shape{100, 100, 100, 100};
    PartialShape expected_out_shape{100, 100, 100, 100};

    std::vector<int32_t> start_val{2, 10, 35, 10};
    std::vector<int32_t> stop_val{8, 25, 40, 20};
    std::vector<int32_t> step_val{1, 1, 1, 100};

    std::vector<int32_t> axes_val{0, 1, 2, 3};

    element::Type_t et = element::f32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};

    EXPECT_THROW(make_slice_op_const_inputs(input_vals, data_shape, et), NodeValidationFailure);
}

TEST(type_prop, slice_v8_input_wrong_shape_catch) {
    PartialShape data_shape{100, 100, 100, 100};

    PartialShape correct_shape{3};
    PartialShape wrong_shape{};

    element::Type_t et = element::i32;

    const auto data = std::make_shared<op::v0::Parameter>(et, data_shape);
    const auto start = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto stop = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto step = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto axes = std::make_shared<op::v0::Parameter>(et, correct_shape);
    {
        try {
            const auto start = std::make_shared<op::v0::Parameter>(et, wrong_shape);
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "`start` input must be a 1D tensor");
        }
    }
    {
        try {
            const auto stop = std::make_shared<op::v0::Parameter>(et, wrong_shape);
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "`stop` input must be a 1D tensor");
        }
    }
    {
        try {
            const auto step = std::make_shared<op::v0::Parameter>(et, wrong_shape);
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "`step` input must be a 1D tensor");
        }
    }
    {
        try {
            const auto axes = std::make_shared<op::v0::Parameter>(et, wrong_shape);
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "`axes` input must be a 1D tensor");
        }
    }
    {
        try {
            const auto data = std::make_shared<op::v0::Parameter>(et, wrong_shape);
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "`data` input can't be a scalar");
        }
    }
}

TEST(type_prop, slice_v8_input_start_stop_step_dif_length_catch) {
    PartialShape data_shape{100, 100, 100, 100};

    PartialShape correct_shape{3};
    PartialShape wrong_shape{2};

    element::Type_t et = element::i32;

    const auto data = std::make_shared<op::v0::Parameter>(et, data_shape);
    const auto start = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto stop = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto step = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto axes = std::make_shared<op::v0::Parameter>(et, correct_shape);
    {
        try {
            const auto start = std::make_shared<op::v0::Parameter>(et, wrong_shape);
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "start`, `stop`, `step` inputs must have compatible shapes");
        }
    }
    {
        try {
            const auto stop = std::make_shared<op::v0::Parameter>(et, wrong_shape);
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "start`, `stop`, `step` inputs must have compatible shapes");
        }
    }
    {
        try {
            const auto step = std::make_shared<op::v0::Parameter>(et, wrong_shape);
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "start`, `stop`, `step` inputs must have compatible shapes");
        }
    }
    {
        try {
            const auto axes = std::make_shared<op::v0::Parameter>(et, wrong_shape);
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(),
                                 "`axes` input must have compatible shape with `start`, `stop`, `step` inputs");
        }
    }
}

TEST(type_prop, slice_v8_input_start_stop_step_out_of_data_rank_length_catch) {
    PartialShape data_shape{100, 100, 100, 100};

    PartialShape correct_shape{3};
    PartialShape wrong_shape{5};

    element::Type_t et = element::i32;

    const auto data = std::make_shared<op::v0::Parameter>(et, data_shape);
    const auto start = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto stop = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto step = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto axes = std::make_shared<op::v0::Parameter>(et, correct_shape);
    {
        try {
            const auto start = std::make_shared<op::v0::Parameter>(et, wrong_shape);
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "`start` input dim size can't be bigger than `data` rank");
        }
    }
    {
        try {
            const auto stop = std::make_shared<op::v0::Parameter>(et, wrong_shape);
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "`stop` input dim size can't be bigger than `data` rank");
        }
    }
    {
        try {
            const auto step = std::make_shared<op::v0::Parameter>(et, wrong_shape);
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "`step` input dim size can't be bigger than `data` rank");
        }
    }
    {
        try {
            const auto axes = std::make_shared<op::v0::Parameter>(et, wrong_shape);
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "`axes` input dim size can't be bigger than `data` rank");
        }
    }
}

TEST(type_prop, slice_v8_input_wrong_types_float_catch) {
    PartialShape data_shape{100, 100, 100, 100};
    PartialShape correct_shape{3};

    element::Type_t et = element::i32;
    element::Type_t wrong_et = element::f32;

    const auto data = std::make_shared<op::v0::Parameter>(et, data_shape);
    const auto start = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto stop = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto step = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto axes = std::make_shared<op::v0::Parameter>(et, correct_shape);
    {
        try {
            const auto start = std::make_shared<op::v0::Parameter>(wrong_et, correct_shape);
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "`start` input type must be integer.");
        }
    }
    {
        try {
            const auto stop = std::make_shared<op::v0::Parameter>(wrong_et, correct_shape);
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "`stop` input type must be integer.");
        }
    }
    {
        try {
            const auto step = std::make_shared<op::v0::Parameter>(wrong_et, correct_shape);
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "`step` input type must be integer.");
        }
    }
    {
        try {
            const auto axes = std::make_shared<op::v0::Parameter>(wrong_et, correct_shape);
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "`axes` input type must be integer.");
        }
    }
}

TEST(type_prop, slice_v8_input_wrong_types_bool_catch) {
    PartialShape data_shape{100, 100, 100, 100};
    PartialShape correct_shape{3};

    element::Type_t et = element::u64;
    element::Type_t wrong_et = element::boolean;

    const auto data = std::make_shared<op::v0::Parameter>(et, data_shape);
    const auto start = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto stop = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto step = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto axes = std::make_shared<op::v0::Parameter>(et, correct_shape);
    {
        try {
            const auto start = std::make_shared<op::v0::Parameter>(wrong_et, correct_shape);
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "`start` input type must be integer.");
        }
    }
    {
        try {
            const auto stop = std::make_shared<op::v0::Parameter>(wrong_et, correct_shape);
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "`stop` input type must be integer.");
        }
    }
    {
        try {
            const auto step = std::make_shared<op::v0::Parameter>(wrong_et, correct_shape);
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "`step` input type must be integer.");
        }
    }
    {
        try {
            const auto axes = std::make_shared<op::v0::Parameter>(wrong_et, correct_shape);
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "`axes` input type must be integer.");
        }
    }
}

TEST(type_prop, slice_v8_basic_const_inputs_out_axes_val) {
    PartialShape data_shape{10, 10, 10, 10, 10, 10, 10, 10};

    std::vector<int32_t> start_val{1, 1, -20, 9, 9, 9, 9, 20};
    std::vector<int32_t> stop_val{8, 8, 20, -11, 0, -10, -11, -20};
    std::vector<int32_t> step_val{1, 2, 1, -1, -1, -1, -2, -1};

    element::Type_t et = element::i32;
    {
        try {
            std::vector<int32_t> axes_val{2, 0, -20, 7, 1, 20, 6, 4};
            std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
            const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);
            FAIL() << "Slice validation did not work!";
        } catch (const ov::AssertFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "must be in range of the `data` input rank: [-8, 7]. Got: -20");
        }
    }
    {
        try {
            std::vector<int32_t> axes_val{2, 0, 9, 7, 1, 20, 6, 4};
            std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
            const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "must be in range of the `data` input rank: [-8, 7]. Got: 9");
        }
    }
    {
        try {
            const auto data = std::make_shared<op::v0::Parameter>(et, data_shape);
            const auto start = std::make_shared<op::v0::Parameter>(et, PartialShape{2});
            const auto stop = std::make_shared<op::v0::Parameter>(et, PartialShape{2});
            const auto step = std::make_shared<op::v0::Parameter>(et, PartialShape{2});
            const auto axes = std::make_shared<op::v0::Constant>(et, Shape{2}, std::vector<int32_t>{-15, 7});
            const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "must be in range of the `data` input rank: [-8, 7]. Got: -15");
        }
    }
}

TEST(type_prop, slice_v8_basic_const_inputs_step_zero) {
    PartialShape data_shape{10, 10, 10, 10, 10, 10, 10, 10};
    PartialShape expected_out_shape{4, 9, 7, 10, 10, 9, 5, 10};

    std::vector<int32_t> start_val{1, 1, -20, 9, 9, 9, 9, 20};
    std::vector<int32_t> stop_val{8, 8, 20, -11, 0, -10, -11, -20};

    element::Type_t et = element::i32;
    {
        std::vector<int32_t> step_val{1, 2, 0, -1, -1, -1, -2, -1};
        std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val};
        try {
            const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);
            FAIL() << "Slice validation did not work!";
        } catch (const NodeValidationFailure& error) {
            EXPECT_HAS_SUBSTRING(error.what(), "'step' value can't be zero");
        }
    }
}

TEST(type_prop, slice_v8_dynamic_rank_inputs) {
    PartialShape dyn_rank_shape = PartialShape::dynamic();
    element::Type_t et = element::i32;

    const auto data = std::make_shared<op::v0::Parameter>(et, dyn_rank_shape);
    const auto start = std::make_shared<op::v0::Parameter>(et, dyn_rank_shape);
    const auto stop = std::make_shared<op::v0::Parameter>(et, dyn_rank_shape);
    const auto step = std::make_shared<op::v0::Parameter>(et, dyn_rank_shape);
    const auto axes = std::make_shared<op::v0::Parameter>(et, dyn_rank_shape);
    const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);

    EXPECT_EQ(op->get_output_partial_shape(0), dyn_rank_shape);
}

TEST(type_prop, slice_dynamic_value_and_label_propagation) {
    Dimension marked_0 = Dimension(3);
    ov::DimensionTracker::set_label(marked_0, 10);
    PartialShape target_0 = PartialShape{marked_0, 4};

    auto param = std::make_shared<op::Parameter>(element::f32, Shape{1});
    auto param_0 = std::make_shared<op::Parameter>(element::f32, target_0);
    auto shape_0 = std::make_shared<op::ShapeOf>(param_0);

    const auto& et = element::i64;
    std::vector<int64_t> start_val{0}, stop_val{1}, step_val{1};
    const auto start = std::make_shared<op::v0::Constant>(et, Shape{start_val.size()}, start_val);
    const auto stop = std::make_shared<op::v0::Constant>(et, Shape{stop_val.size()}, stop_val);
    const auto step = std::make_shared<op::v0::Constant>(et, Shape{step_val.size()}, step_val);
    const auto slice = std::make_shared<op::v8::Slice>(shape_0, start, stop, step);

    auto bc = std::make_shared<op::v1::Broadcast>(param, slice);
    ASSERT_EQ(bc->get_shape(), (Shape{3}));

    const auto& output_shape = bc->get_output_partial_shape(0);
    ASSERT_EQ(ov::DimensionTracker::get_label(output_shape[0]), 10);
}
