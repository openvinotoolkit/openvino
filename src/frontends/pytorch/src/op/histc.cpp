// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

using namespace ov::op;

namespace ov::frontend::pytorch::op {

OutputVector translate_histc(const NodeContext& context) {
    num_inputs_check(context, 4, 5);
    auto input = context.get_input(0);
    auto bins = context.const_input<int64_t>(1);
    auto min_val = context.const_input<double>(2);
    auto max_val = context.const_input<double>(3);

    // Ensure input is floating point
    if (!input.get_element_type().is_real()) {
        input = context.mark_node(std::make_shared<v0::Convert>(input, element::f32));
    }

    // Flatten input to 1D
    auto minus_one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto flat_input = context.mark_node(std::make_shared<v1::Reshape>(input, minus_one, false));

    // Compute min/max values
    Output<Node> computed_min, computed_max;
    if (min_val == 0 && max_val == 0) {
        auto axes = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
        computed_min = context.mark_node(std::make_shared<v1::ReduceMin>(flat_input, axes, false));
        computed_max = context.mark_node(std::make_shared<v1::ReduceMax>(flat_input, axes, false));
    } else {
        auto input_type = flat_input->get_element_type();
        computed_min = context.mark_node(v0::Constant::create(input_type, Shape{}, {static_cast<float>(min_val)}));
        computed_max = context.mark_node(v0::Constant::create(input_type, Shape{}, {static_cast<float>(max_val)}));
    }

    // Calculate bin width
    auto bin_diff = context.mark_node(std::make_shared<v1::Subtract>(computed_max, computed_min));
    auto bins_node =
        context.mark_node(v0::Constant::create(bin_diff->get_element_type(), Shape{}, {static_cast<float>(bins)}));
    auto bin_width = context.mark_node(std::make_shared<v1::Divide>(bin_diff, bins_node));

    // Handle zero bin_diff case
    auto zero = context.mark_node(v0::Constant::create(bin_diff->get_element_type(), Shape{}, {0}));
    auto bin_diff_zero = context.mark_node(std::make_shared<v1::Equal>(bin_diff, zero));
    auto safe_bin_width = context.mark_node(std::make_shared<v1::Select>(
        bin_diff_zero,
        context.mark_node(v0::Constant::create(bin_diff->get_output_element_type(0), Shape{}, {1.0f})),
        bin_width));

    // Calculate bin indices
    auto shifted = context.mark_node(std::make_shared<v1::Subtract>(flat_input, computed_min));
    auto bin_idx_float = context.mark_node(std::make_shared<v1::Divide>(shifted, safe_bin_width));
    auto bin_idx_floor = context.mark_node(std::make_shared<v0::Floor>(bin_idx_float));

    // Clamp indices
    auto zero_i64 = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    auto max_bin = context.mark_node(v0::Constant::create(element::i64, Shape{}, {bins - 1}));
    auto bin_idx_clamped = context.mark_node(std::make_shared<v0::Clamp>(bin_idx_floor, zero_i64, max_bin));
    auto bin_idx = context.mark_node(std::make_shared<v0::Convert>(bin_idx_clamped, element::i64));

    // Create mask for valid elements
    auto ge_min = context.mark_node(std::make_shared<v1::GreaterEqual>(flat_input, computed_min));
    auto le_max = context.mark_node(std::make_shared<v1::LessEqual>(flat_input, computed_max));
    auto valid_mask = context.mark_node(std::make_shared<v1::LogicalAnd>(ge_min, le_max));
    auto mask = context.mark_node(std::make_shared<v0::Convert>(valid_mask, element::f32));

    // Prepare updates
    auto ones = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1.0}));
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(flat_input, element::i64));
    auto updates = context.mark_node(std::make_shared<v3::Broadcast>(ones, input_shape));
    auto masked_updates = context.mark_node(std::make_shared<v1::Multiply>(updates, mask));

    // Initialize and scatter
    auto hist_shape = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {bins}));
    auto zero_f = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0.0}));
    auto histogram = context.mark_node(std::make_shared<v3::Broadcast>(zero_f, hist_shape));
    auto axis = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    auto scatter =
        context.mark_node(std::make_shared<v12::ScatterElementsUpdate>(histogram,
                                                                       bin_idx,
                                                                       masked_updates,
                                                                       axis,
                                                                       v12::ScatterElementsUpdate::Reduction::SUM));

    if (!context.input_is_none(4)) {
        context.mutate_input(4, scatter);
    }

    return {scatter};
}

}  // namespace ov::frontend::pytorch::op