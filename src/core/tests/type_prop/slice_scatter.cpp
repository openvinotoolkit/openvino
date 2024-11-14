// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/slice_scatter.hpp"

#include <gtest/gtest.h>

#include <tuple>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "sequence_generator.hpp"

namespace ov {
namespace test {

using ov::op::v0::Constant;
using ov::op::v0::Parameter;

class TypePropSliceScatterTest : public TypePropOpTest<op::v15::SliceScatter> {
public:
    template <typename T>
    std::shared_ptr<Node> make_op_const_inputs(const std::vector<std::vector<T>>& args,
                                               PartialShape& data_shape,
                                               PartialShape& updates_shape,
                                               element::Type_t et) {
        const auto& start_val = args[0];
        const auto& stop_val = args[1];
        const auto& step_val = args[2];

        const auto data = std::make_shared<Parameter>(et, data_shape);
        const auto updates = std::make_shared<Parameter>(et, updates_shape);
        const auto start = std::make_shared<Constant>(et, Shape{start_val.size()}, start_val);
        const auto stop = std::make_shared<Constant>(et, Shape{stop_val.size()}, stop_val);
        const auto step = std::make_shared<Constant>(et, Shape{step_val.size()}, step_val);

        if (args.size() > 3) {
            const auto& axes_val = args[3];
            const auto axes = std::make_shared<Constant>(et, Shape{axes_val.size()}, axes_val);
            return make_op(data, updates, start, stop, step, axes);
        }
        return make_op(data, updates, start, stop, step);
    }
};

TEST_F(TypePropSliceScatterTest, default_ctor) {
    const auto data = std::make_shared<Parameter>(element::f32, Shape{5, 5, 5});
    const auto updates = std::make_shared<Parameter>(element::f32, Shape{5, 2, 5});
    const auto start = std::make_shared<Constant>(element::i32, Shape{2}, std::vector<int64_t>{0, 1});
    const auto stop = std::make_shared<Constant>(element::i32, Shape{2}, std::vector<int64_t>{25, 5});
    const auto step = std::make_shared<Constant>(element::i32, Shape{2}, std::vector<int64_t>{1, 2});
    const auto op = make_op();
    op->set_arguments(OutputVector{data, updates, start, stop, step});
    op->validate_and_infer_types();
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 5);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({5, 5, 5}));
}

TEST_F(TypePropSliceScatterTest, default_ctor_axes) {
    const auto data = std::make_shared<Parameter>(element::f64, Shape{5, 5, 5});
    const auto updates = std::make_shared<Parameter>(element::f64, Shape{2, 5, 5});
    const auto start = std::make_shared<Constant>(element::i64, Shape{2}, std::vector<int64_t>{0, 1});
    const auto stop = std::make_shared<Constant>(element::i64, Shape{2}, std::vector<int64_t>{25, 5});
    const auto step = std::make_shared<Constant>(element::i64, Shape{2}, std::vector<int64_t>{1, 2});
    const auto axes = std::make_shared<Constant>(element::i32, Shape{2}, std::vector<int64_t>{2, 0});
    const auto op = make_op();
    op->set_arguments(OutputVector{data, updates, start, stop, step, axes});
    op->validate_and_infer_types();
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 6);
    EXPECT_EQ(op->get_output_element_type(0), element::f64);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({5, 5, 5}));
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs) {
    PartialShape data_shape{10, 10, 10, 10, 10, 10, 10, 10};
    PartialShape updates_shape{7, 4, 10, 10, 9, 9, 5, 10};

    std::vector<int32_t> start_val{1, 1, -20, 9, 9, 9, 9, 20};
    std::vector<int32_t> stop_val{8, 8, 20, -11, 0, -10, -11, -20};
    std::vector<int32_t> step_val{1, 2, 1, -1, -1, -1, -2, -1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    constexpr auto et = element::i32;

    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_dif_steps) {
    PartialShape data_shape{5, 5, 5, 5, 9, 9};
    PartialShape updates_shape{3, 2, 2, 1, 5, 3};

    std::vector<int32_t> start_val{0, 0, 0, 0, 0, 0};
    std::vector<int32_t> stop_val{5, 5, 5, 5, 9, 9};
    std::vector<int32_t> step_val{2, 3, 4, 5, 2, 3};

    constexpr auto et = element::u8;

    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_dif_neg_steps) {
    PartialShape data_shape{5, 5, 5, 5, 9, 9};
    PartialShape updates_shape{2, 2, 1, 1, 4, 3};

    std::vector<int32_t> start_val{5, 5, 5, 5, 9, 9};
    std::vector<int32_t> stop_val{0, 0, 0, 0, 0, 0};
    std::vector<int32_t> step_val{-2, -3, -4, -5, -2, -3};

    constexpr auto et = element::i8;

    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_default_axes) {
    PartialShape data_shape{10, 10, 10, 10, 10, 10, 10, 10};
    PartialShape updates_shape{7, 4, 10, 10, 9, 9, 5, 10};

    std::vector<int32_t> start_val{1, 1, -20, 9, 9, 9, 9, 20};
    std::vector<int32_t> stop_val{8, 8, 20, -11, 0, -10, -11, -20};
    std::vector<int32_t> step_val{1, 2, 1, -1, -1, -1, -2, -1};

    constexpr auto et = element::i16;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, const_inputs_output_zero_dims) {
    PartialShape data_shape{5, 5, 5, 5, 9, 9};
    PartialShape updates_shape{3, 0, 5, 0, 5, 0};

    std::vector<int32_t> start_val{0, 0, 0, 5, 0, 10};
    std::vector<int32_t> stop_val{5, 5, 5, 5, 9, 9};
    std::vector<int32_t> step_val{2, -3, 1, 5, 2, 1};

    constexpr auto et = element::i32;

    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_unordered_axes) {
    PartialShape data_shape{10, 10, 10, 10, 10, 10, 10, 10};
    PartialShape updates_shape{4, 9, 7, 10, 10, 9, 5, 10};

    std::vector<int32_t> start_val{1, 1, -20, 9, 9, 9, 9, 20};
    std::vector<int32_t> stop_val{8, 8, 20, -11, 0, -10, -11, -20};
    std::vector<int32_t> step_val{1, 2, 1, -1, -1, -1, -2, -1};

    std::vector<int32_t> axes_val{2, 0, 3, 7, 1, 5, 6, 4};

    constexpr auto et = element::i32;

    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, const_inputs_not_all_axes_unordered_prop_symbols) {
    PartialShape data_shape{10, 10, 10, 10, 10, 20, Dimension(20, 30), 30, Dimension(2, 5), Dimension(-1)};
    PartialShape updates_shape{4, 7, 10, 10, 9, 20, Dimension(10, 15), 30, Dimension(2, 5), Dimension(-1)};

    auto symbols = set_shape_symbols(data_shape);

    std::vector<int32_t> start_val{1, 1, -20, 9, 10, 9};
    std::vector<int32_t> stop_val{8, 8, 20, -11, 25, 0};
    std::vector<int32_t> step_val{1, 2, 1, -1, 1, -1};

    std::vector<int32_t> axes_val{1, 0, 2, 3, 6, 4};

    constexpr auto et = element::i32;

    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
    EXPECT_EQ(get_shape_symbols(op->get_output_partial_shape(0)), symbols);
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_data_dynamic_bounds_dimensions) {
    PartialShape data_shape{Dimension(1, 5), Dimension(10, 20), Dimension(20, 30), 16, Dimension(30, 40)};
    PartialShape updates_shape{1, 6, Dimension(10, 15), 16, Dimension(0, 5)};

    std::vector<int32_t> start_val{0, 2, 10, 0, 35};
    std::vector<int32_t> stop_val{1, 8, 25, 16, 40};
    std::vector<int32_t> step_val{1, 1, 1, 1, 1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    constexpr auto et = element::u64;

    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_data_dynamic_rank) {
    PartialShape data_shape = PartialShape::dynamic();
    PartialShape updates_shape = PartialShape::dynamic();

    std::vector<int32_t> start_val{1, 1, -20, 9, 9, 9, 9, 20};
    std::vector<int32_t> stop_val{8, 8, 20, -11, 0, -10, -11, -20};
    std::vector<int32_t> step_val{1, 2, 1, -1, -1, -1, -2, -1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    constexpr auto et = element::i32;

    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
    EXPECT_TRUE(op->get_output_partial_shape(0).rank().is_dynamic());
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_MAX_MIN_INT_dynamic_dimensions) {
    PartialShape data_shape{10,
                            Dimension(10, 20),
                            Dimension(10, 20),
                            Dimension(10, 20),
                            Dimension(10, 20),
                            Dimension(10, 20),
                            Dimension(30, 40),
                            Dimension(0, 50),
                            Dimension(0, -1)};
    PartialShape updates_shape{8,
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

    constexpr auto et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_MAX_MIN_INT_dynamic_dimensions_neg_step) {
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
    PartialShape updates_shape{8,
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

    constexpr auto et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_data_full_dynamic_dims) {
    PartialShape data_shape{-1, -1, -1, -1};
    PartialShape updates_shape{{0, 6}, {0, 15}, {0, 5}, -1};

    std::vector<int32_t> start_val{2, 10, 35, INT32_MIN};
    std::vector<int32_t> stop_val{8, 25, 40, -3};
    std::vector<int32_t> step_val{1, 1, 1, 1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    constexpr auto et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_data_full_dynamic_dims_neg_ind) {
    PartialShape data_shape{Dimension(-1), Dimension(-1), Dimension(-1)};
    PartialShape updates_shape{Dimension(0, 6), Dimension(0, 15), Dimension(0, 5)};

    std::vector<int32_t> start_val{-8, -25, -40};
    std::vector<int32_t> stop_val{-2, -10, -35};
    std::vector<int32_t> step_val{1, 1, 1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    constexpr auto et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_data_full_dynamic_dims_neg_step) {
    PartialShape data_shape{Dimension(-1), Dimension(-1), Dimension(-1)};
    PartialShape updates_shape{Dimension(0, 6), Dimension(0, 15), Dimension(0, 5)};

    std::vector<int32_t> start_val{8, 25, 40};
    std::vector<int32_t> stop_val{2, 10, 35};
    std::vector<int32_t> step_val{-1, -1, -1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    constexpr auto et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_data_full_dynamic_dims_neg_step_neg_ind) {
    PartialShape data_shape{Dimension(-1), Dimension(-1), Dimension(-1)};
    PartialShape updates_shape{Dimension(0, 6), Dimension(0, 15), Dimension(0, 5)};

    std::vector<int32_t> start_val{-2, -10, -35};
    std::vector<int32_t> stop_val{-8, -25, -40};
    std::vector<int32_t> step_val{-1, -1, -1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    constexpr auto et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_data_full_dynamic_dims_neg_step_mix_ind) {
    PartialShape data_shape{-1, -1, -1, -1, -1, -1};
    PartialShape updates_shape{{0, 6}, {0, 3}, {0, 6}, -1, -1, -1};

    std::vector<int32_t> start_val{5, 5, 5, -10, INT32_MAX, INT32_MAX};
    std::vector<int32_t> stop_val{-10, -10, INT32_MIN, 5, 5, INT32_MIN};
    std::vector<int32_t> step_val{-1, -2, -1, -1, -1, -1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    constexpr auto et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_dynamic_dims_maxint32) {
    PartialShape data_shape{Dimension(0, 2000), Dimension(-1), 4};
    PartialShape updates_shape{Dimension(0, 2000), Dimension(-1), Dimension(4)};

    std::vector<int32_t> start_val{0, 0, 0};
    std::vector<int32_t> stop_val{INT32_MAX, INT32_MAX, INT32_MAX};
    std::vector<int32_t> step_val{1, 1, 1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    constexpr auto et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_dynamic_dims_maxint64) {
    PartialShape data_shape{Dimension(0, 2000), Dimension(-1), 4};
    PartialShape updates_shape{Dimension(0, 2000), Dimension(-1), Dimension(4)};

    std::vector<int64_t> start_val{0, 0, 0};
    std::vector<int64_t> stop_val{INT64_MAX, INT64_MAX, INT64_MAX};
    std::vector<int64_t> step_val{1, 1, 1};

    std::vector<int64_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    constexpr auto et = element::i64;
    std::vector<std::vector<int64_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_dynamic_dims_maxint32_start1) {
    PartialShape data_shape{Dimension(0, 2000), Dimension(-1), 4};
    PartialShape updates_shape{Dimension(0, 2000), Dimension(-1), Dimension(4)};

    std::vector<int32_t> start_val{1};
    std::vector<int32_t> stop_val{INT32_MAX};
    std::vector<int32_t> step_val{1};

    std::vector<int32_t> axes_val{1};

    constexpr auto et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_dynamic_dims_maxint64_start1) {
    PartialShape data_shape{Dimension(0, 2000), Dimension(-1), 4};
    PartialShape updates_shape{Dimension(0, 2000), Dimension(-1), Dimension(4)};

    std::vector<int64_t> start_val{1};
    std::vector<int64_t> stop_val{INT64_MAX};
    std::vector<int64_t> step_val{1};

    std::vector<int64_t> axes_val{1};

    constexpr auto et = element::i64;
    std::vector<std::vector<int64_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_MAX_MIN_INT_64_dynamic_dimensions_neg_step) {
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
    PartialShape updates_shape{8,
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

    constexpr auto et = element::i64;
    std::vector<std::vector<int64_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_MAX_MIN_64_no_upper_bounds_neg_step) {
    PartialShape data_shape{Dimension(-1), Dimension(0, INT64_MAX), Dimension(0, INT64_MAX)};
    PartialShape updates_shape{Dimension(-1), Dimension(0, INT64_MAX), Dimension(0, INT64_MAX)};

    std::vector<int64_t> start_val{INT64_MAX, INT64_MAX, INT64_MAX};
    std::vector<int64_t> stop_val{INT64_MIN, INT64_MIN, INT64_MIN};
    std::vector<int64_t> step_val{-1, -1, -1};

    std::vector<int64_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    constexpr auto et = element::i64;
    std::vector<std::vector<int64_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_MAX_MIN_64_no_upper_bounds) {
    PartialShape data_shape{Dimension(-1), Dimension(0, INT64_MAX), Dimension(0, INT64_MAX)};
    PartialShape updates_shape{Dimension(-1), Dimension(0, INT64_MAX), Dimension(0, INT64_MAX)};

    std::vector<int64_t> start_val{INT64_MIN, INT64_MIN, INT64_MIN};
    std::vector<int64_t> stop_val{INT64_MAX, INT64_MAX, INT64_MAX};
    std::vector<int64_t> step_val{1, 1, 1};

    std::vector<int64_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    constexpr auto et = element::i64;
    std::vector<std::vector<int64_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_MAX_MIN_32_no_upper_bounds) {
    PartialShape data_shape{Dimension(-1), Dimension(0, INT64_MAX), Dimension(0, INT32_MAX), Dimension(0, INT32_MAX)};
    PartialShape updates_shape{Dimension(-1),
                               Dimension(0, INT64_MAX),
                               Dimension(0, INT32_MAX),
                               Dimension(0, INT32_MAX)};

    std::vector<int32_t> start_val{INT32_MIN, INT32_MIN, INT32_MIN, INT32_MIN};
    std::vector<int32_t> stop_val{INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX};
    std::vector<int32_t> step_val{1, 1, 1, 1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    constexpr auto et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, basic_const_inputs_MAX_MIN_32_no_upper_bounds_neg_step) {
    PartialShape data_shape{Dimension(-1), Dimension(0, INT64_MAX), Dimension(0, INT32_MAX), Dimension(0, INT32_MAX)};
    PartialShape updates_shape{Dimension(-1), Dimension(-1), Dimension(0, INT32_MAX), Dimension(0, INT32_MAX)};

    std::vector<int32_t> start_val{INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX};
    std::vector<int32_t> stop_val{INT32_MIN, INT32_MIN, INT32_MIN, INT32_MIN};
    std::vector<int32_t> step_val{-1, -1, -1, -1};

    std::vector<int32_t> axes_val(start_val.size());
    std::iota(axes_val.begin(), axes_val.end(), 0);

    constexpr auto et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, dynamic_dim_zero_start_negative_stop) {
    PartialShape data_shape{Dimension(-1)};
    PartialShape updates_shape{Dimension(-1)};

    std::vector<int32_t> start_val{0};
    std::vector<int32_t> stop_val{-2};
    std::vector<int32_t> step_val{1};

    std::vector<int32_t> axes_val{0};

    constexpr auto et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};
    const auto op = make_op_const_inputs(input_vals, data_shape, updates_shape, et);

    EXPECT_EQ(op->get_element_type(), et);
    EXPECT_EQ(op->get_output_partial_shape(0), data_shape);
}

TEST_F(TypePropSliceScatterTest, duplicated_axes) {
    PartialShape data_shape{100, 100, 100, 100};
    PartialShape updates_shape{100, 100, 100, 100};

    std::vector<int32_t> start_val{2, 10, 35, 10};
    std::vector<int32_t> stop_val{8, 25, 40, 20};
    std::vector<int32_t> step_val{1, 1, 1, 100};

    std::vector<int32_t> axes_val{2, 1, 2, 3};

    constexpr auto et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};

    EXPECT_THROW(make_op_const_inputs(input_vals, data_shape, updates_shape, et), NodeValidationFailure);
}

TEST_F(TypePropSliceScatterTest, zero_step) {
    PartialShape data_shape{100, 100, 100};
    PartialShape updates_shape{100, 100, 100};

    std::vector<int32_t> start_val{2, 10, 35};
    std::vector<int32_t> stop_val{8, 25, 40};
    std::vector<int32_t> step_val{1, 0, 1};

    std::vector<int32_t> axes_val{1, 2, 3};

    constexpr auto et = element::i32;
    std::vector<std::vector<int32_t>> input_vals{start_val, stop_val, step_val, axes_val};

    EXPECT_THROW(make_op_const_inputs(input_vals, data_shape, updates_shape, et), NodeValidationFailure);
}

TEST_F(TypePropSliceScatterTest, all_params_dynamic) {
    const auto data = std::make_shared<Parameter>(element::boolean, PartialShape::dynamic());
    const auto updates = std::make_shared<Parameter>(element::boolean, PartialShape::dynamic());
    const auto start = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto stop = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto step = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto op = make_op(data, updates, start, stop, step);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 5);
    EXPECT_EQ(op->get_output_element_type(0), element::boolean);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST_F(TypePropSliceScatterTest, all_params_dynamic_dynamic_types) {
    const auto data = std::make_shared<Parameter>(element::dynamic, PartialShape::dynamic());
    const auto updates = std::make_shared<Parameter>(element::dynamic, PartialShape::dynamic());
    const auto start = std::make_shared<Parameter>(element::dynamic, PartialShape::dynamic());
    const auto stop = std::make_shared<Parameter>(element::dynamic, PartialShape::dynamic());
    const auto step = std::make_shared<Parameter>(element::dynamic, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::dynamic, PartialShape::dynamic());
    const auto op = make_op(data, updates, start, stop, step, axes);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 6);
    EXPECT_EQ(op->get_output_element_type(0), element::dynamic);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST_F(TypePropSliceScatterTest, all_params_dynamic_with_axes) {
    const auto data = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto updates = std::make_shared<Parameter>(element::f16, PartialShape::dynamic());
    const auto start = std::make_shared<Parameter>(element::u16, PartialShape::dynamic());
    const auto stop = std::make_shared<Parameter>(element::u16, PartialShape::dynamic());
    const auto step = std::make_shared<Parameter>(element::u16, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i64, PartialShape::dynamic());
    const auto op = make_op(data, updates, start, stop, step, axes);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 6);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST_F(TypePropSliceScatterTest, all_params_dynamic_merge_rank) {
    const auto data = std::make_shared<Parameter>(element::bf16, PartialShape::dynamic());
    const auto updates = std::make_shared<Parameter>(element::bf16, PartialShape::dynamic(Dimension(5)));
    const auto start = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto stop = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto step = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::u8, PartialShape::dynamic());
    const auto op = make_op(data, updates, start, stop, step, axes);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 6);
    EXPECT_EQ(op->get_output_element_type(0), element::bf16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic(Dimension(5)));
}

TEST_F(TypePropSliceScatterTest, all_params_static) {
    const auto data = std::make_shared<Parameter>(element::i32, PartialShape{5, 5, 5});
    const auto updates = std::make_shared<Parameter>(element::i32, PartialShape{5, 1, 2});
    const auto start = std::make_shared<Parameter>(element::i32, PartialShape{3});
    const auto stop = std::make_shared<Parameter>(element::i32, PartialShape{3});
    const auto step = std::make_shared<Parameter>(element::i32, PartialShape{3});
    const auto op = make_op(data, updates, start, stop, step);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 5);
    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({5, 5, 5}));
}

TEST_F(TypePropSliceScatterTest, all_params_static_types_dynamic) {
    const auto data = std::make_shared<Parameter>(element::dynamic, PartialShape{5, 5, 5});
    const auto updates = std::make_shared<Parameter>(element::dynamic, PartialShape{5, 1, 2});
    const auto start = std::make_shared<Parameter>(element::dynamic, PartialShape{3});
    const auto stop = std::make_shared<Parameter>(element::dynamic, PartialShape{3});
    const auto step = std::make_shared<Parameter>(element::dynamic, PartialShape{3});
    const auto axes = std::make_shared<Parameter>(element::dynamic, PartialShape{3});
    const auto op = make_op(data, updates, start, stop, step, axes);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 6);
    EXPECT_EQ(op->get_output_element_type(0), element::dynamic);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({5, 5, 5}));
}

TEST_F(TypePropSliceScatterTest, all_params_static_with_axes) {
    const auto data = std::make_shared<Parameter>(element::i64, PartialShape{5, 5, 5});
    const auto updates = std::make_shared<Parameter>(element::i64, PartialShape{5, 1, 2});
    const auto start = std::make_shared<Parameter>(element::i32, PartialShape{3});
    const auto stop = std::make_shared<Parameter>(element::i32, PartialShape{3});
    const auto step = std::make_shared<Parameter>(element::i32, PartialShape{3});
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{3});
    const auto op = make_op(data, updates, start, stop, step, axes);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_input_size(), 6);
    EXPECT_EQ(op->get_output_element_type(0), element::i64);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape({5, 5, 5}));
}

TEST_F(TypePropSliceScatterTest, incompatible_input_size) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape{5, 5, 5});
    const auto updates = std::make_shared<Parameter>(element::f32, PartialShape{5, 1, 2});
    const auto start = std::make_shared<Parameter>(element::i32, PartialShape{3});
    const auto stop = std::make_shared<Parameter>(element::i32, PartialShape{3});
    const auto op = make_op();
    op->set_arguments(OutputVector{data, updates, start, stop});
    OV_EXPECT_THROW(op->validate_and_infer_types(),
                    NodeValidationFailure,
                    testing::HasSubstr("SliceScatter has to have 5 or 6 inputs. Got: 4"));
}

TEST_F(TypePropSliceScatterTest, incompatible_data_updates_shape_scalar) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto updates = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto start = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto stop = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto step = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto scalar = std::make_shared<Parameter>(element::f32, PartialShape{});
    {
        OV_EXPECT_THROW(std::ignore = make_op(scalar, updates, start, stop, step, axes),
                        NodeValidationFailure,
                        testing::HasSubstr("SliceScatter `data` and `updates` input can't be a scalar."));
    }
    {
        OV_EXPECT_THROW(std::ignore = make_op(data, scalar, start, stop, step, axes),
                        NodeValidationFailure,
                        testing::HasSubstr("SliceScatter `data` and `updates` input can't be a scalar."));
    }
}

TEST_F(TypePropSliceScatterTest, incompatible_updates_shape_rank) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(2));
    const auto updates = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
    const auto start = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto stop = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto step = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    OV_EXPECT_THROW(std::ignore = make_op(data, updates, start, stop, step, axes),
                    NodeValidationFailure,
                    testing::HasSubstr("SliceScatter `data` and `updates` need to have compatible rank."));
}

TEST_F(TypePropSliceScatterTest, incompatible_updates_shape_expected_slice_const) {
    const auto data =
        std::make_shared<Parameter>(element::f32, PartialShape{5, 5, 5, Dimension::dynamic(), Dimension::dynamic()});
    const auto updates =
        std::make_shared<Parameter>(element::f32, PartialShape{5, 2, 3, Dimension::dynamic(), Dimension::dynamic()});
    {
        const auto start = std::make_shared<Constant>(element::i32, Shape{2}, std::vector<int64_t>{25, -25});
        const auto stop = std::make_shared<Constant>(element::i32, Shape{2}, std::vector<int64_t>{125, -1});
        const auto step = std::make_shared<Constant>(element::i32, Shape{2}, std::vector<int64_t>{1, 1});
        const auto axes = std::make_shared<Constant>(element::i32, Shape{2}, std::vector<int64_t>{0, -4});
        OV_EXPECT_THROW(
            std::ignore = make_op(data, updates, start, stop, step, axes),
            NodeValidationFailure,
            testing::HasSubstr(
                "SliceScatter updates at index 1 are not compatible with expected slice shape [0,4,5,?,?]"));
    }
    {
        const auto start = std::make_shared<Constant>(element::i32, Shape{3}, std::vector<int64_t>{25, -125, 5});
        const auto stop = std::make_shared<Constant>(element::i32, Shape{3}, std::vector<int64_t>{125, 5, 5});
        const auto step = std::make_shared<Constant>(element::i32, Shape{3}, std::vector<int64_t>{1, -1, 15});
        const auto axes = std::make_shared<Constant>(element::i32, Shape{3}, std::vector<int64_t>{0, -4, 2});
        OV_EXPECT_THROW(
            std::ignore = make_op(data, updates, start, stop, step, axes),
            NodeValidationFailure,
            testing::HasSubstr(
                "SliceScatter updates at index 1 are not compatible with expected slice shape [0,0,0,?,?]"));
    }
    {
        const auto start = std::make_shared<Constant>(element::i32, Shape{2}, std::vector<int64_t>{25, -25});
        const auto stop = std::make_shared<Constant>(element::i32, Shape{2}, std::vector<int64_t>{125, -1});
        const auto step = std::make_shared<Constant>(element::i32, Shape{2}, std::vector<int64_t>{1, 1});
        const auto axes = std::make_shared<Constant>(element::i32, Shape{2}, std::vector<int64_t>{0, -4});
        OV_EXPECT_THROW(
            std::ignore = make_op(data, updates, start, stop, step, axes),
            NodeValidationFailure,
            testing::HasSubstr(
                "SliceScatter updates at index 1 are not compatible with expected slice shape [0,4,5,?,?]"));
    }
    {
        const auto start = std::make_shared<Constant>(element::i32, Shape{2}, std::vector<int64_t>{25, -25});
        const auto stop = std::make_shared<Constant>(element::i32, Shape{2}, std::vector<int64_t>{125, -1});
        const auto step = std::make_shared<Constant>(element::i32, Shape{2}, std::vector<int64_t>{1, 1});
        OV_EXPECT_THROW(
            std::ignore = make_op(data, updates, start, stop, step),
            NodeValidationFailure,
            testing::HasSubstr(
                "SliceScatter updates at index 1 are not compatible with expected slice shape [0,4,5,?,?]"));
    }
}

TEST_F(TypePropSliceScatterTest, incompatible_slice_shapes) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto updates = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto start = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto stop = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto step = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto incorrect_shape = std::make_shared<Parameter>(element::i32, PartialShape{2});
    {
        OV_EXPECT_THROW(std::ignore = make_op(data, updates, incorrect_shape, stop, step, axes),
                        NodeValidationFailure,
                        testing::HasSubstr("SliceScatter `start`, `stop`, `step` inputs must have compatible shapes."));
    }
    {
        OV_EXPECT_THROW(std::ignore = make_op(data, updates, start, incorrect_shape, step, axes),
                        NodeValidationFailure,
                        testing::HasSubstr("SliceScatter `start`, `stop`, `step` inputs must have compatible shapes."));
    }
    {
        OV_EXPECT_THROW(std::ignore = make_op(data, updates, start, stop, incorrect_shape, axes),
                        NodeValidationFailure,
                        testing::HasSubstr("SliceScatter `start`, `stop`, `step` inputs must have compatible shapes."));
    }
    {
        OV_EXPECT_THROW(
            std::ignore = make_op(data, updates, start, stop, step, incorrect_shape),
            NodeValidationFailure,
            testing::HasSubstr(
                "SliceScatter `axes` input must have compatible shape with `start`, `stop`, `step` inputs."));
    }
}

TEST_F(TypePropSliceScatterTest, incompatible_slice_rank) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto updates = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto start = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto stop = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto step = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto incorrect_rank = std::make_shared<Parameter>(element::i32, PartialShape{});
    {
        OV_EXPECT_THROW(std::ignore = make_op(data, updates, incorrect_rank, stop, step, axes),
                        NodeValidationFailure,
                        testing::HasSubstr("SliceScatter `start` input must be a 1D tensor. Got rank: 0"));
    }
    {
        OV_EXPECT_THROW(std::ignore = make_op(data, updates, start, incorrect_rank, step, axes),
                        NodeValidationFailure,
                        testing::HasSubstr("SliceScatter `stop` input must be a 1D tensor. Got rank: 0"));
    }
    {
        OV_EXPECT_THROW(std::ignore = make_op(data, updates, start, stop, incorrect_rank, axes),
                        NodeValidationFailure,
                        testing::HasSubstr("SliceScatter `step` input must be a 1D tensor. Got rank: 0"));
    }
    {
        OV_EXPECT_THROW(std::ignore = make_op(data, updates, start, stop, step, incorrect_rank),
                        NodeValidationFailure,
                        testing::HasSubstr("SliceScatter `axes` input must be a 1D tensor. Got rank: 0"));
    }
}

TEST_F(TypePropSliceScatterTest, incompatible_slice_length_data_rank) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
    const auto updates = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto start = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto stop = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto step = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto incorrect_rank = std::make_shared<Parameter>(element::i32, PartialShape{4});
    {
        OV_EXPECT_THROW(
            std::ignore = make_op(data, updates, incorrect_rank, stop, step, axes),
            NodeValidationFailure,
            testing::HasSubstr("SliceScatter `start` input dim size can't be bigger than `data` or `updates` rank."));
    }
    {
        OV_EXPECT_THROW(
            std::ignore = make_op(data, updates, start, incorrect_rank, step, axes),
            NodeValidationFailure,
            testing::HasSubstr("SliceScatter `stop` input dim size can't be bigger than `data` or `updates` rank."));
    }
    {
        OV_EXPECT_THROW(
            std::ignore = make_op(data, updates, start, stop, incorrect_rank, axes),
            NodeValidationFailure,
            testing::HasSubstr("SliceScatter `step` input dim size can't be bigger than `data` or `updates` rank."));
    }
    {
        OV_EXPECT_THROW(
            std::ignore = make_op(data, updates, start, stop, step, incorrect_rank),
            NodeValidationFailure,
            testing::HasSubstr("SliceScatter `axes` input dim size can't be bigger than `data` or `updates` rank."));
    }
}

TEST_F(TypePropSliceScatterTest, incompatible_slice_length_updates_rank) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto updates =
        std::make_shared<Parameter>(element::f32,
                                    PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()});
    const auto start = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto stop = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto step = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto incorrect_rank = std::make_shared<Parameter>(element::i32, PartialShape{4});
    {
        OV_EXPECT_THROW(
            std::ignore = make_op(data, updates, incorrect_rank, stop, step, axes),
            NodeValidationFailure,
            testing::HasSubstr("SliceScatter `start` input dim size can't be bigger than `data` or `updates` rank."));
    }
    {
        OV_EXPECT_THROW(
            std::ignore = make_op(data, updates, start, incorrect_rank, step, axes),
            NodeValidationFailure,
            testing::HasSubstr("SliceScatter `stop` input dim size can't be bigger than `data` or `updates` rank."));
    }
    {
        OV_EXPECT_THROW(
            std::ignore = make_op(data, updates, start, stop, incorrect_rank, axes),
            NodeValidationFailure,
            testing::HasSubstr("SliceScatter `step` input dim size can't be bigger than `data` or `updates` rank."));
    }
    {
        OV_EXPECT_THROW(
            std::ignore = make_op(data, updates, start, stop, step, incorrect_rank),
            NodeValidationFailure,
            testing::HasSubstr("SliceScatter `axes` input dim size can't be bigger than `data` or `updates` rank."));
    }
}

TEST_F(TypePropSliceScatterTest, incorrect_slice_values) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic(3));
    const auto updates = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto start = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto stop = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto step = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    {
        const auto incorrect_step_val = std::make_shared<Constant>(element::i32, Shape{1}, std::vector<int64_t>{0});
        OV_EXPECT_THROW(std::ignore = make_op(data, updates, start, stop, incorrect_step_val, axes),
                        NodeValidationFailure,
                        testing::HasSubstr("SliceScatter step values must be non-zero."));
    }
    {
        const auto duplicated_axes = std::make_shared<Constant>(element::i32, Shape{2}, std::vector<int64_t>{0, 0});
        OV_EXPECT_THROW(std::ignore = make_op(data, updates, start, stop, step, duplicated_axes),
                        NodeValidationFailure,
                        testing::HasSubstr("SliceScatter values in `axes` input must be unique."));
    }
}

TEST_F(TypePropSliceScatterTest, incompatible_slice_param_types) {
    const auto data = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto updates = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    const auto start = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto stop = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto step = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto axes = std::make_shared<Parameter>(element::i32, PartialShape::dynamic());
    const auto incorrect_rank = std::make_shared<Parameter>(element::f32, PartialShape::dynamic());
    {
        OV_EXPECT_THROW(std::ignore = make_op(data, updates, incorrect_rank, stop, step, axes),
                        NodeValidationFailure,
                        testing::HasSubstr("SliceScatter `start` input type must be integer."));
    }
    {
        OV_EXPECT_THROW(std::ignore = make_op(data, updates, start, incorrect_rank, step, axes),
                        NodeValidationFailure,
                        testing::HasSubstr("SliceScatter `stop` input type must be integer."));
    }
    {
        OV_EXPECT_THROW(std::ignore = make_op(data, updates, start, stop, incorrect_rank, axes),
                        NodeValidationFailure,
                        testing::HasSubstr("SliceScatter `step` input type must be integer."));
    }
    {
        OV_EXPECT_THROW(std::ignore = make_op(data, updates, start, stop, step, incorrect_rank),
                        NodeValidationFailure,
                        testing::HasSubstr("SliceScatter `axes` input type must be integer."));
    }
}
}  // namespace test
}  // namespace ov
