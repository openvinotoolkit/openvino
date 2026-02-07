// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
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
    int64_t bins = context.const_input<int64_t>(1);
    double min_val = context.const_input<double>(2);
    double max_val = context.const_input<double>(3);

    // Flatten input and convert into f64 
    auto flat_shape = v0::Constant::create(element::i64, Shape{1}, {-1});
    auto flat_input = context.mark_node(std::make_shared<v1::Reshape>(input, flat_shape, false));
    auto f64_input = context.mark_node(std::make_shared<v0::Convert>(flat_input, element::f64));

    // calculate bin width: (max - min) / bins
    auto min_const = v0::Constant::create(element::f64, Shape{}, {min_val});
    auto max_const = v0::Constant::create(element::f64, Shape{}, {max_val});
    auto bins_const = v0::Constant::create(element::f64, Shape{}, {static_cast<double>(bins)});
    auto range = context.mark_node(std::make_shared<v1::Subtract>(max_const, min_const));
    auto bin_width = context.mark_node(std::make_shared<v1::Divide>(range, bins_const));

    // calculate bin indexes: floor((x - min) / bin_width)
    auto shift = context.mark_node(std::make_shared<v1::Subtract>(f64_input, min_const));
    auto normalized = context.mark_node(std::make_shared<v1::Divide>(shift, bin_width));
    auto floored = context.mark_node(std::make_shared<v0::Floor>(normalized));
    auto bin_idxs = context.mark_node(std::make_shared<v0::Convert>(floored, element::i64));
    
    // Indexes should be in valid range
    auto bin_idx_range = context.mark_node(
        std::make_shared<v0::Clamp>(bin_idxs, 0.0, static_cast<double>(bins - 1)));

    // Init histogram with zeros
    auto zero_const = v0::Constant::create(element::f64, Shape{}, {0.0});
    auto bins_shape = v0::Constant::create(element::i64, Shape{1}, {bins});
    auto histogram = context.mark_node(std::make_shared<v3::Broadcast>(zero_const, bins_shape));

    // create ones for counting elements
    auto one_const = v0::Constant::create(element::f64, Shape{}, {1.0});
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(flat_input, element::i64));
    auto histogram_ones = context.mark_node(std::make_shared<v3::Broadcast>(one_const, input_shape));

    // Count elements per bin using ScatterElementsUpdate with SUM reduction
    auto axis = v0::Constant::create(element::i64, Shape{}, {0});
    auto histogram_res = context.mark_node(std::make_shared<v12::ScatterElementsUpdate>(
        histogram, bin_idx_range, histogram_ones, axis,
        v12::ScatterElementsUpdate::Reduction::SUM));

    // type check
    auto dtype_input = input.get_element_type();
    auto dtype_res = dtype_input.is_static() ? dtype_input : element::f32;
    
    return {context.mark_node(std::make_shared<v0::Convert>(histogram_res, dtype_res))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
