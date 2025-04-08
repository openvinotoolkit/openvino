// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/opsets/opset9.hpp"
#include "sequence_generator.hpp"

using namespace ov;
using namespace testing;

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

TEST(type_prop, slice_v8_const_inputs_not_all_axes_unordered_prop_symbols) {
    PartialShape data_shape{10, 10, 10, 10, 10, 20, Dimension(20, 30), 30, Dimension(2, 5), Dimension(-1)};
    PartialShape expected_out_shape{4, 7, 10, 10, 9, 20, Dimension(10, 15), 30, Dimension(2, 5), Dimension(-1)};

    auto symbols = set_shape_symbols(data_shape);

    std::vector<int32_t> start_val{1, 1, -20, 9, 10, 9};
    std::vector<int32_t> stop_val{8, 8, 20, -11, 25, 0};
    std::vector<int32_t> step_val{1, 2, 1, -1, 1, -1};

    std::vector<int32_t> axes_val{1, 0, 2, 3, 6, 4};

    constexpr auto et = element::i32;

    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                ElementsAre(nullptr,
                            nullptr,
                            symbols[2],
                            symbols[3],
                            nullptr,
                            symbols[5],
                            nullptr,
                            symbols[7],
                            symbols[8],
                            symbols[9]));
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

TEST(type_prop, slice_v8_basic_param_inputs_default_axes_symbols_prop) {
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
    auto symbols = set_shape_symbols(data_shape);

    PartialShape start_shape{7};
    PartialShape stop_shape{7};
    PartialShape step_shape{7};

    constexpr auto et = element::i32;

    const auto data = std::make_shared<op::v0::Parameter>(et, data_shape);
    const auto start = std::make_shared<op::v0::Parameter>(et, start_shape);
    const auto stop = std::make_shared<op::v0::Parameter>(et, stop_shape);
    const auto step = std::make_shared<op::v0::Parameter>(et, step_shape);

    const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), expected_out_shape);
    EXPECT_THAT(
        get_shape_symbols(op->get_output_partial_shape(0)),
        ElementsAre(symbols[0], nullptr, nullptr, nullptr, nullptr, nullptr, symbols[6], symbols[7], symbols[8]));
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
    PartialShape data_shape{-1, -1, -1, -1};
    PartialShape expected_out_shape{{0, 6}, {0, 15}, {0, 5}, -1};

    std::vector<int32_t> start_val{2, 10, 35, INT32_MIN};
    std::vector<int32_t> stop_val{8, 25, 40, -3};
    std::vector<int32_t> step_val{1, 1, 1, 1};

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
    PartialShape data_shape{-1, -1, -1, -1, -1, -1};
    PartialShape expected_out_shape{{0, 6}, {0, 3}, {0, 6}, -1, -1, -1};

    std::vector<int32_t> start_val{5, 5, 5, -10, INT32_MAX, INT32_MAX};
    std::vector<int32_t> stop_val{-10, -10, INT32_MIN, 5, 5, INT32_MIN};
    std::vector<int32_t> step_val{-1, -2, -1, -1, -1, -1};

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
                                    Dimension(4, 9),
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
    std::vector<int64_t> step_val{-1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

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
    const auto wrong_shape_in = std::make_shared<op::v0::Parameter>(et, wrong_shape);

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(data, wrong_shape_in, stop, step, axes),
                    NodeValidationFailure,
                    HasSubstr("`start` input must be a 1D tensor"));

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(data, start, wrong_shape_in, step, axes),
                    NodeValidationFailure,
                    HasSubstr("`stop` input must be a 1D tensor"));

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(data, start, stop, wrong_shape_in, axes),
                    NodeValidationFailure,
                    HasSubstr("`step` input must be a 1D tensor"));

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, wrong_shape_in),
                    NodeValidationFailure,
                    HasSubstr("`axes` input must be a 1D tensor"));

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(wrong_shape_in, start, stop, step, axes),
                    NodeValidationFailure,
                    HasSubstr("`data` input can't be a scalar"));
}

TEST(type_prop, slice_v8_input_start_stop_step_dif_length_catch) {
    PartialShape data_shape{100, 100, 100, 100};

    PartialShape correct_shape{3};
    PartialShape wrong_shape{2};

    constexpr auto et = element::i32;

    const auto data = std::make_shared<op::v0::Parameter>(et, data_shape);
    const auto start = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto stop = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto step = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto axes = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto wrong_shape_in = std::make_shared<op::v0::Parameter>(et, wrong_shape);

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(data, wrong_shape_in, stop, step, axes),
                    NodeValidationFailure,
                    HasSubstr("start`, `stop`, `step` inputs must have compatible shapes"));

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(data, start, wrong_shape_in, step, axes),
                    NodeValidationFailure,
                    HasSubstr("start`, `stop`, `step` inputs must have compatible shapes"));

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(data, start, stop, wrong_shape_in, axes),
                    NodeValidationFailure,
                    HasSubstr("start`, `stop`, `step` inputs must have compatible shapes"));

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, wrong_shape_in),
                    NodeValidationFailure,
                    HasSubstr("`axes` input must have compatible shape with `start`, `stop`, `step` inputs"));
}

TEST(type_prop, slice_v8_input_start_stop_step_out_of_data_rank_length_catch) {
    PartialShape data_shape{100, 100, 100, 100};

    PartialShape correct_shape{3};
    PartialShape wrong_shape{5};

    constexpr auto et = element::i32;

    const auto data = std::make_shared<op::v0::Parameter>(et, data_shape);
    const auto start = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto stop = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto step = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto axes = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto wrong_shape_in = std::make_shared<op::v0::Parameter>(et, wrong_shape);

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(data, wrong_shape_in, stop, step, axes),
                    NodeValidationFailure,
                    HasSubstr("`start` input dim size can't be bigger than `data` rank"));

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(data, start, wrong_shape_in, step, axes),
                    NodeValidationFailure,
                    HasSubstr("`stop` input dim size can't be bigger than `data` rank"));

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(data, start, stop, wrong_shape_in, axes),
                    NodeValidationFailure,
                    HasSubstr("`step` input dim size can't be bigger than `data` rank"));

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, wrong_shape_in),
                    NodeValidationFailure,
                    HasSubstr("`axes` input dim size can't be bigger than `data` rank"));
}

TEST(type_prop, slice_v8_input_wrong_types_float_catch) {
    PartialShape data_shape{100, 100, 100, 100};
    PartialShape correct_shape{3};

    constexpr auto et = element::i32;
    constexpr auto wrong_et = element::f32;

    const auto data = std::make_shared<op::v0::Parameter>(et, data_shape);
    const auto start = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto stop = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto step = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto axes = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto wrong_et_shape = std::make_shared<op::v0::Parameter>(wrong_et, correct_shape);

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(data, wrong_et_shape, stop, step, axes),
                    NodeValidationFailure,
                    HasSubstr("`start` input type must be integer."));

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(data, start, wrong_et_shape, step, axes),
                    NodeValidationFailure,
                    HasSubstr("`stop` input type must be integer."));

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(data, start, stop, wrong_et_shape, axes),
                    NodeValidationFailure,
                    HasSubstr("`step` input type must be integer."));

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, wrong_et_shape),
                    NodeValidationFailure,
                    HasSubstr("`axes` input type must be integer."));
}

TEST(type_prop, slice_v8_input_wrong_types_bool_catch) {
    PartialShape data_shape{100, 100, 100, 100};
    PartialShape correct_shape{3};

    constexpr auto et = element::u64;
    constexpr auto wrong_et = element::boolean;

    const auto data = std::make_shared<op::v0::Parameter>(et, data_shape);
    const auto start = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto stop = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto step = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto axes = std::make_shared<op::v0::Parameter>(et, correct_shape);
    const auto wrong_et_shape = std::make_shared<op::v0::Parameter>(wrong_et, correct_shape);

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(data, wrong_et_shape, stop, step, axes),
                    NodeValidationFailure,
                    HasSubstr("`start` input type must be integer."));

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(data, start, wrong_et_shape, step, axes),
                    NodeValidationFailure,
                    HasSubstr("`stop` input type must be integer."));

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(data, start, stop, wrong_et_shape, axes),
                    NodeValidationFailure,
                    HasSubstr("`step` input type must be integer."));

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, wrong_et_shape),
                    NodeValidationFailure,
                    HasSubstr("`axes` input type must be integer."));
}

TEST(type_prop, slice_v8_basic_const_inputs_out_axes_val) {
    PartialShape data_shape{10, 10, 10, 10, 10, 10, 10, 10};

    std::vector<int32_t> start_val{1, 1, -20, 9, 9, 9, 9, 20};
    std::vector<int32_t> stop_val{8, 8, 20, -11, 0, -10, -11, -20};
    std::vector<int32_t> step_val{1, 2, 1, -1, -1, -1, -2, -1};

    constexpr auto et = element::i32;
    {
        std::vector<int32_t> axes_val{2, 0, -20, 7, 1, 20, 6, 4};
        std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
        OV_EXPECT_THROW(const auto op = make_slice_op_const_inputs(input_vals, data_shape, et),
                        NodeValidationFailure,
                        HasSubstr("Axis -20 out of the tensor rank range [-8, 7]"));
    }
    {
        std::vector<int32_t> axes_val{2, 0, 9, 7, 1, 20, 6, 4};
        std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
        OV_EXPECT_THROW(const auto op = make_slice_op_const_inputs(input_vals, data_shape, et),
                        NodeValidationFailure,
                        HasSubstr("Axis 9 out of the tensor rank range [-8, 7]"));
    }

    const auto data = std::make_shared<op::v0::Parameter>(et, data_shape);
    const auto start = std::make_shared<op::v0::Parameter>(et, PartialShape{2});
    const auto stop = std::make_shared<op::v0::Parameter>(et, PartialShape{2});
    const auto step = std::make_shared<op::v0::Parameter>(et, PartialShape{2});
    const auto axes = std::make_shared<op::v0::Constant>(et, Shape{2}, std::vector<int32_t>{-15, 7});

    OV_EXPECT_THROW(const auto op = std::make_shared<op::v8::Slice>(data, start, stop, step, axes),
                    NodeValidationFailure,
                    HasSubstr("Axis -15 out of the tensor rank range [-8, 7]"));
}

TEST(type_prop, slice_v8_basic_const_inputs_step_zero) {
    PartialShape data_shape{10, 10, 10, 10, 10, 10, 10, 10};
    PartialShape expected_out_shape{4, 9, 7, 10, 10, 9, 5, 10};

    std::vector<int32_t> start_val{1, 1, -20, 9, 9, 9, 9, 20};
    std::vector<int32_t> stop_val{8, 8, 20, -11, 0, -10, -11, -20};
    std::vector<int32_t> step_val{1, 2, 0, -1, -1, -1, -2, -1};
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val};

    OV_EXPECT_THROW(const auto op = make_slice_op_const_inputs(input_vals, data_shape, element::i32),
                    NodeValidationFailure,
                    HasSubstr("Step must be non-zero"));
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

TEST(type_prop, slice_v8_dynamic_value_and_symbol_propagation) {
    Dimension marked_0 = Dimension(3, 7);
    auto symbol = std::make_shared<Symbol>();
    marked_0.set_symbol(symbol);
    PartialShape target_0 = PartialShape{marked_0, 4};

    auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto param_0 = std::make_shared<ov::op::v0::Parameter>(element::f32, target_0);
    auto shape_0 = std::make_shared<op::v0::ShapeOf>(param_0);

    const auto& et = element::i64;
    std::vector<int64_t> start_val{0}, stop_val{1}, step_val{1};
    const auto start = std::make_shared<op::v0::Constant>(et, Shape{start_val.size()}, start_val);
    const auto stop = std::make_shared<op::v0::Constant>(et, Shape{stop_val.size()}, stop_val);
    const auto step = std::make_shared<op::v0::Constant>(et, Shape{step_val.size()}, step_val);
    const auto slice = std::make_shared<op::v8::Slice>(shape_0, start, stop, step);

    auto bc = std::make_shared<op::v1::Broadcast>(param, slice);

    const auto& output_shape = bc->get_output_partial_shape(0);
    EXPECT_EQ(output_shape, (PartialShape{{3, 7}}));
    EXPECT_EQ(output_shape[0].get_symbol(), symbol);
}

TEST(type_prop, slice_v8_dynamic_dimension_but_slice_min_is_lt_input_min_size) {
    PartialShape data_shape{Dimension(20, -1)};

    std::vector<int32_t> start_val{-7};
    std::vector<int32_t> stop_val{INT32_MAX};
    std::vector<int32_t> step_val{1};
    std::vector<int32_t> axes_val{0};

    constexpr auto et = element::i32;

    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_slice_op_const_inputs(input_vals, data_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({{7}}));
}

TEST(type_prop, slice_v8_use_default_ctor) {
    const auto zero_mask = std::vector<int64_t>(3, 0);

    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{10, 11, 12, 2});
    auto start = ov::op::v0::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
    auto stop = ov::op::v0::Constant::create(element::i64, Shape{4}, {1, 5, 20, 20});
    auto step = ov::op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1});

    auto slice = std::make_shared<op::v8::Slice>();
    slice->set_arguments(ov::OutputVector{data, start, stop, step});
    slice->validate_and_infer_types();

    ASSERT_EQ(slice->get_output_partial_shape(0), PartialShape({1, 5, 12, 2}));
}

TEST(type_prop, slice_v8_stop_is_shape_of_with_bounds) {
    auto shape = PartialShape{1, {5, 7}};
    set_shape_symbols(shape);
    const auto p_stop = std::make_shared<ov::op::v0::Parameter>(element::i64, shape);
    const auto shape_of_stop = std::make_shared<op::v0::ShapeOf>(p_stop);

    auto data = ov::op::v0::Constant::create(element::i64, Shape{1, 10}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 0});
    auto start = ov::op::v0::Constant::create(element::i64, Shape{2}, {0, 0});
    auto steps = ov::op::v0::Constant::create(element::i64, Shape{2}, {1, 1});

    auto slice = std::make_shared<op::v8::Slice>(data, start, shape_of_stop, steps);

    EXPECT_EQ(slice->get_output_partial_shape(0), PartialShape({1, {5, 7}}));
    EXPECT_THAT(get_shape_symbols(slice->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, slice_v8_start_is_shape_of_with_bounds) {
    auto shape = PartialShape{0, {3, 5}};
    set_shape_symbols(shape);
    const auto p_start = std::make_shared<ov::op::v0::Parameter>(element::i64, shape);
    const auto shape_of_start = std::make_shared<op::v0::ShapeOf>(p_start);

    auto data = ov::op::v0::Constant::create(element::i64, Shape{1, 10}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 0});
    auto stop = ov::op::v0::Constant::create(element::i64, Shape{2}, {1, 7});
    auto steps = ov::op::v0::Constant::create(element::i64, Shape{2}, {1, 1});

    auto slice = std::make_shared<op::v8::Slice>(data, shape_of_start, stop, steps);

    EXPECT_EQ(slice->get_output_partial_shape(0), PartialShape({1, {2, 4}}));
    EXPECT_THAT(get_shape_symbols(slice->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, slice_v8_start_stop_is_shape_of_with_bounds) {
    auto start_shape = PartialShape{0, {3, 5}};
    auto stop_shape = PartialShape{2, {6, 7}};
    set_shape_symbols(start_shape);
    set_shape_symbols(stop_shape);
    const auto p_start = std::make_shared<ov::op::v0::Parameter>(element::i64, start_shape);
    const auto p_stop = std::make_shared<ov::op::v0::Parameter>(element::i64, stop_shape);
    const auto shape_of_start = std::make_shared<op::v0::ShapeOf>(p_start);
    const auto shape_of_stop = std::make_shared<op::v0::ShapeOf>(p_stop);

    auto data = ov::op::v0::Constant::create(element::i64, Shape{1, 10}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 0});
    auto steps = ov::op::v0::Constant::create(element::i64, Shape{2}, {1, 1});

    auto slice = std::make_shared<op::v8::Slice>(data, shape_of_start, shape_of_stop, steps);

    EXPECT_EQ(slice->get_output_partial_shape(0), PartialShape({1, {1, 4}}));
    EXPECT_THAT(get_shape_symbols(slice->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, slice_v8_unknowns_axes) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{5, 10, 15});
    const auto start = std::make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{-1});
    const auto stop = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{1});
    const auto steps = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{1});
    const auto axes = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{1});

    auto slice = std::make_shared<op::v8::Slice>(data, start, stop, steps, axes);

    EXPECT_EQ(slice->get_output_partial_shape(0), PartialShape({{0, 5}, {0, 10}, {0, 15}}));
}

TEST(type_prop, slice_v8_inf_dim_start_from_last_N_to_end) {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, 256, -1});
    auto start = ov::op::v0::Constant::create(element::i64, Shape{1}, {-7});
    auto stop = ov::op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{INT64_MAX});
    auto step = ov::op::v0::Constant::create(element::i64, Shape{1}, {1});
    auto axes = ov::op::v0::Constant::create(element::i64, Shape{1}, {2});

    auto slice = std::make_shared<op::v8::Slice>(data, start, stop, step, axes);

    EXPECT_EQ(slice->get_output_partial_shape(0), PartialShape({1, 256, {0, 7}}));
}

using SliceV8IntervalParams =
    std::tuple<ov::PartialShape, ov::PartialShape, ov::PartialShape, int64_t, int64_t, int64_t, ov::PartialShape>;

class SliceV8IntervalTest : public TypePropOpTest<op::v8::Slice>, public WithParamInterface<SliceV8IntervalParams> {
protected:
    void SetUp() override {
        std::tie(data_shape, start_shape, stop_shape, start_offset, stop_offset, step, exp_shape) = GetParam();
    }

    ov::PartialShape data_shape, start_shape, stop_shape, exp_shape;
    int64_t start_offset, stop_offset, step;
};

INSTANTIATE_TEST_SUITE_P(type_prop,
                         SliceV8IntervalTest,
                         Values(SliceV8IntervalParams({1024}, {{0, 20}}, {{10, 20}}, 0, 0, 1, {{0, 20}}),
                                SliceV8IntervalParams({1024}, {{0, 20}}, {{10, 20}}, 0, 0, 1, {{0, 20}}),
                                SliceV8IntervalParams({-1}, {{0, 20}}, {{10, 20}}, 10, 0, 1, {{0, 20}}),
                                SliceV8IntervalParams({1024}, {{0, 10}}, {{0, 5}}, 0, 10, 1, {{1004, 1019}}),
                                SliceV8IntervalParams({{120, 1024}}, {{0, 10}}, {{0, 5}}, 0, 10, 1, {{100, 1019}}),
                                SliceV8IntervalParams({1024}, {{0, 1030}}, {{0, 2000}}, 1025, 10, 1, {{0, 1024}}),
                                SliceV8IntervalParams({1024}, {{1, 12}}, {{0, 18}}, 10, 0, 2, {{0, 9}}),
                                SliceV8IntervalParams({1024}, {{10, 20}}, {{0, 20}}, 0, 10, 2, {{0, 507}}),
                                SliceV8IntervalParams({{100, 1024}}, {{10, 20}}, {{0, 20}}, 0, 10, 2, {{0, 507}}),
                                SliceV8IntervalParams({1024}, {10}, {30}, 0, 0, 2, {10}),
                                SliceV8IntervalParams({{20, 1024}}, {{10, 15}}, {{30, 40}}, 0, 0, 2, {{3, 15}}),
                                // reverse stride
                                SliceV8IntervalParams({1024}, {{0, 20}}, {{10, 20}}, 10, 0, -1, {{0, 1013}}),
                                SliceV8IntervalParams({-1}, {{0, 20}}, {{10, 20}}, 10, 0, -1, {-1}),
                                SliceV8IntervalParams({1024}, {30}, {10}, 35, 40, -1, {25}),
                                SliceV8IntervalParams({1024}, {{0, 2000}}, {{0, 1030}}, 10, 1026, -1, {{0, 1024}}),
                                SliceV8IntervalParams({1024}, {30}, {10}, 0, 0, -2, {10}),
                                SliceV8IntervalParams({1024}, {{20, 30}}, {10}, 0, 0, -2, {{5, 10}}),
                                SliceV8IntervalParams({1024}, {{20, 30}}, {10}, 40, 0, -2, {{497, 502}}),
                                SliceV8IntervalParams({1024}, {{20, 30}}, {{10, 15}}, 0, 0, -2, {{3, 10}}),
                                SliceV8IntervalParams({{10, 1024}}, {{20, 30}}, {{10, 15}}, 0, 0, -2, {{0, 10}})));

TEST_P(SliceV8IntervalTest, start_stop_as_interval) {
    using namespace ov::opset9;

    const auto p_start = std::make_shared<Parameter>(element::i64, start_shape);
    const auto shape_of_start = std::make_shared<ShapeOf>(p_start);
    const auto start =
        std::make_shared<Subtract>(shape_of_start, Constant::create(element::i64, Shape{1}, {start_offset}));

    const auto p_stop = std::make_shared<Parameter>(element::i64, stop_shape);
    const auto shape_of_stop = std::make_shared<ShapeOf>(p_stop);
    const auto stop =
        std::make_shared<Subtract>(shape_of_stop, Constant::create(element::i64, Shape{1}, {stop_offset}));

    const auto data = std::make_shared<Parameter>(element::f32, data_shape);
    const auto steps = Constant::create(element::i64, Shape{1}, {step});

    const auto op = make_op(data, start, stop, steps);

    EXPECT_EQ(op->get_output_partial_shape(0), exp_shape);
}
