// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "pt_framework_node.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_quantize_per_tensor(const NodeContext& context) {
    num_inputs_check(context, 4, 4);
    const auto input = context.get_input(0);
    const auto scale = context.get_input(1);
    const auto zero_point = context.get_input(2);
    const auto dtype = convert_dtype(context.const_input<int64_t>(3));
    return {quantize(context, input, scale, zero_point, dtype, QuantizedPtNodeType::QUANTIZE_PER_TENSOR)};
}

OutputVector translate_quantize_per_channel(const NodeContext& context) {
    num_inputs_check(context, 5, 5);
    const auto input = context.get_input(0);
    const auto scales = context.get_input(1);
    const auto zero_points = context.get_input(2);
    const auto axis = context.get_input(3);
    const auto dtype = convert_dtype(context.const_input<int64_t>(4));
    return {quantize(context, input, scales, zero_points, axis, dtype, QuantizedPtNodeType::QUANTIZE_PER_CHANNEL)};
}

OutputVector translate_quantize_per_tensor_fx(const NodeContext& context) {
    num_inputs_check(context, 4, 8);
    const auto input = context.get_input(0);
    const auto scale = context.get_input(1);
    const auto zero_point = context.get_input(2);
    auto low = context.const_input<int64_t>(3);
    auto high = context.const_input<int64_t>(4);
    return {quantize_fx(context,
                        input,
                        scale,
                        zero_point,
                        low,
                        high,
                        element::i8,
                        QuantizedPtNodeType::QUANTIZE_PER_TENSOR)};
}

OutputVector translate_quantize_per_channel_fx(const NodeContext& context) {
    num_inputs_check(context, 4, 8);
    const auto input = context.get_input(0);
    const auto scales = context.get_input(1);
    const auto zero_points = context.get_input(2);
    const auto axis = context.get_input(3);
    auto low = context.const_input<int64_t>(4);
    auto high = context.const_input<int64_t>(5);
    return {quantize_fx(context,
                        input,
                        scales,
                        zero_points,
                        axis,
                        low,
                        high,
                        element::i8,
                        QuantizedPtNodeType::QUANTIZE_PER_CHANNEL)};
}

OutputVector translate_fake_quantize_per_tensor_affine_fx(const NodeContext& context) {
    num_inputs_check(context, 5, 6);

    const auto input = context.get_input(0);
    const auto scale = context.get_input(1);
    const auto zero_point = context.get_input(2);
    const auto num_inputs = context.get_input_size();
    OutputVector res;
    if (num_inputs == 5) {
        const auto low = context.const_input<int64_t>(3);
        const auto high = context.const_input<int64_t>(4);
        const auto quantized = quantize_fx(context,
                                           input,
                                           scale,
                                           zero_point,
                                           Output<Node>{},
                                           low,
                                           high,
                                           element::f32,
                                           QuantizedPtNodeType::QUANTIZE_PER_TENSOR);
        res = {quantized};
    } else {
        const auto fake_quant_enabled = context.const_input<bool>(3);
        PYTORCH_OP_CONVERSION_CHECK(fake_quant_enabled == true, "Disabled fake_quant is not supported.");
        const auto low = context.const_input<int64_t>(4);
        const auto high = context.const_input<int64_t>(5);
        const auto quantized = quantize_fx(context,
                                           input,
                                           scale,
                                           zero_point,
                                           Output<Node>{},
                                           low,
                                           high,
                                           element::f32,
                                           QuantizedPtNodeType::QUANTIZE_PER_TENSOR);
        // Returning cache mask is not supported
        auto cachemask = std::make_shared<PtFrameworkNode>(context.get_decoder(), context.inputs(), 1, false, true);
        context.mark_node(cachemask);
        auto attrs = cachemask->get_attrs();
        attrs[PtFrameworkNode::failed_conversion_key] = "Cache mask is not supported for fake_quantize.";
        cachemask->set_attrs(attrs);
        res = {quantized, cachemask};
    }
    return {context.mark_node(make_list_construct(OutputVector{res}))};
}

OutputVector translate_fake_quantize_per_channel_affine_fx(const NodeContext& context) {
    num_inputs_check(context, 6, 6);

    const auto input = context.get_input(0);
    const auto scale = context.get_input(1);
    const auto zero_point = context.get_input(2);
    const auto axis = context.get_input(3);
    const auto low = context.const_input<int64_t>(4);
    const auto high = context.const_input<int64_t>(5);
    const auto res = quantize_fx(context,
                                 input,
                                 scale,
                                 zero_point,
                                 axis,
                                 low,
                                 high,
                                 element::f32,
                                 QuantizedPtNodeType::QUANTIZE_PER_CHANNEL);
    return {context.mark_node(make_list_construct(OutputVector{res}))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
