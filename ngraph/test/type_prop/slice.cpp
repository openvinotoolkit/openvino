// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

template <typename T>
std::shared_ptr<Node> make_slice_op_param_inputs(const std::vector<PartialShape>& args,
                                                 PartialShape& data_shape,
                                                 element::Type_t et) {
    const auto& start_shape = args[0];
    const auto& stop_shape = args[1];
    const auto& step_shape = args[2];

    const auto data = std::make_shared<op::v0::Parameter>(et, data_shape);
    const auto start = std::make_shared<op::v0::Parameter>(et, start_shape);
    const auto stop = std::make_shared<op::v0::Parameter>(et, stop_shape);
    const auto step = std::make_shared<op::v0::Parameter>(et, step_shape);

    if (args.size() > 3) {
        const auto& axes_shape = args[3];
        const auto axes = std::make_shared<op::Parameter>(et, axes_shape);
        return std::make_shared<op::v8::Slice>(data, start, stop, step, axes);
    }
    return std::make_shared<op::v8::Slice>(data, start, stop, step);
}
}  // namespace

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
