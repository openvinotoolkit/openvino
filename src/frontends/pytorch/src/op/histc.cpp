// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/greater_equal.hpp"
#include "openvino/op/less_equal.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_histc(const NodeContext& context) {
    num_inputs_check(context, 1, 4);
    auto input = context.get_input(0);
    int64_t bins = 100;
    double min_val = 0.0;
    double max_val = 0.0;
    if (!context.input_is_none(1))
        bins = context.const_input<int64_t>(1);
    if (!context.input_is_none(2))
        min_val = context.const_input<double>(2);
    if (!context.input_is_none(3))
        max_val = context.const_input<double>(3);

    FRONT_END_OP_CONVERSION_CHECK(bins > 0,
                                  "Pytorch frontend: histc expects 'bins' to be a positive integer, got ",
                                  bins);
    // Flatten and cast to f64
    auto flat_shape = v0::Constant::create(element::i64, Shape{1}, {-1});
    auto flat_input = context.mark_node(std::make_shared<v1::Reshape>(input, flat_shape, false));
    auto f64_input = context.mark_node(std::make_shared<v0::Convert>(flat_input, element::f64));

    // If min==max==0 (PyTorch default), derive range from data; otherwise use constants
    Output<Node> min_node, max_node;
    if (min_val == 0.0 && max_val == 0.0) {
        auto axes = v0::Constant::create(element::i64, Shape{1}, {0});
        min_node = context.mark_node(std::make_shared<v1::ReduceMin>(f64_input, axes, false));
        max_node = context.mark_node(std::make_shared<v1::ReduceMax>(f64_input, axes, false));
    } else {
        min_node = v0::Constant::create(element::f64, Shape{}, {min_val});
        max_node = v0::Constant::create(element::f64, Shape{}, {max_val});
    }

    auto zero_f64   = v0::Constant::create(element::f64, Shape{}, {0.0});
    auto one_f64    = v0::Constant::create(element::f64, Shape{}, {1.0});
    auto bins_const = v0::Constant::create(element::f64, Shape{}, {static_cast<double>(bins)});

    // bin_width = (max - min) / bins; guard against zero range to avoid division by zero
    auto range = context.mark_node(std::make_shared<v1::Subtract>(max_node, min_node));
    auto range_is_zero = context.mark_node(std::make_shared<v1::Equal>(range, zero_f64));
    auto safe_range = context.mark_node(std::make_shared<v1::Select>(range_is_zero, one_f64, range));
    auto bin_width = context.mark_node(std::make_shared<v1::Divide>(safe_range, bins_const));

    // Compute bin index: floor((x - min) / bin_width).
    // Clamp in float domain before converting to i64 to avoid UB from out-of-range casts.
    auto shift = context.mark_node(std::make_shared<v1::Subtract>(f64_input, min_node));
    auto floored = context.mark_node(std::make_shared<v0::Floor>(
        context.mark_node(std::make_shared<v1::Divide>(shift, bin_width))));
    auto clamped_f =
        context.mark_node(std::make_shared<v0::Clamp>(floored, 0.0, static_cast<double>(bins - 1)));
    // Mask out-of-range elements so they contribute 0 to the histogram and to avoid NaN-to-integer casts.
    auto in_range = context.mark_node(std::make_shared<v1::LogicalAnd>(
        context.mark_node(std::make_shared<v1::GreaterEqual>(f64_input, min_node)),
        context.mark_node(std::make_shared<v1::LessEqual>(f64_input, max_node))));
    auto clamped_f_safe =
        context.mark_node(std::make_shared<v1::Select>(in_range, clamped_f, zero_f64));
    auto bin_idxs =
        context.mark_node(std::make_shared<v0::Convert>(clamped_f_safe, element::i64));

    // When range is zero, put all elements in the middle bin (PyTorch behaviour)
    auto input_size = context.mark_node(std::make_shared<v3::ShapeOf>(flat_input, element::i64));
    auto mid_bin = context.mark_node(std::make_shared<v3::Broadcast>(
        v0::Constant::create(element::i64, Shape{}, {bins / 2}), input_size));
    auto clamped_idxs =
        context.mark_node(std::make_shared<v1::Select>(range_is_zero, mid_bin, bin_idxs));
    auto ones = context.mark_node(std::make_shared<v3::Broadcast>(one_f64, input_size));
    auto updates = context.mark_node(std::make_shared<v1::Select>(in_range, ones, zero_f64));

    // Scatter-add into zero-initialised histogram
    auto histogram = context.mark_node(std::make_shared<v3::Broadcast>(
        v0::Constant::create(element::f64, Shape{}, {0.0}),
        v0::Constant::create(element::i64, Shape{1}, {bins})));
    auto axis = v0::Constant::create(element::i64, Shape{}, {0});
    auto histogram_res = context.mark_node(std::make_shared<v12::ScatterElementsUpdate>(
        histogram, clamped_idxs, updates, axis, v12::ScatterElementsUpdate::Reduction::SUM));

    auto dtype_res = input.get_element_type().is_static() ? input.get_element_type() : element::f32;
    return {context.mark_node(std::make_shared<v0::Convert>(histogram_res, dtype_res))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
