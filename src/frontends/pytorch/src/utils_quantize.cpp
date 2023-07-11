// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils_quantize.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

using namespace ov::op;

std::shared_ptr<ov::Node> quantize(const NodeContext& context, ov::Output<ov::Node>& input, std::shared_ptr<ov::Node>& quantized_node) {

    std::shared_ptr<QuantizedPtNode> quantized_pt_node;
    if ((quantized_pt_node = cast_quantized_fw_node(quantized_node, QuantizedPtNode::quantize_per_tensor))) {
        const auto input = quantized_pt_node->get_input_node_ptr(0);
        const auto dtype = quantized_node->get_input_element_type(0);
        const auto scale = quantized_pt_node->get_scale();
        const auto zero_point = quantized_pt_node->get_zero_point();

        const auto scale_convert = context.mark_node(std::make_shared<v1::ConvertLike>(scale, input));
        const auto zero_point_convert = context.mark_node(std::make_shared<v1::ConvertLike>(zero_point, input));
        const auto scaled_input = context.mark_node(std::make_shared<v1::Divide>(input, scale_convert));
        const auto scaled_input_with_zero_pt = context.mark_node(std::make_shared<v1::Add>(scaled_input, zero_point_convert));
        const auto quantized_input =
            context.mark_node(std::make_shared<v5::Round>(scaled_input_with_zero_pt, v5::Round::RoundMode::HALF_TO_EVEN));
        
        ov::Output<ov::Node> output;
        if (dtype == element::u8) {
            const auto clamp = context.mark_node(std::make_shared<v0::Clamp>(quantized_input,
                                                        std::numeric_limits<unsigned char>::lowest(),
                                                        std::numeric_limits<unsigned char>::max()));
            output = context.mark_node(std::make_shared<v0::Convert>(clamp, element::u8));
        } else if (dtype == element::i8) {
            const auto clamp = context.mark_node(std::make_shared<v0::Clamp>(quantized_input,
                                                        std::numeric_limits<unsigned char>::lowest(),
                                                        std::numeric_limits<unsigned char>::max()));
            output = context.mark_node(std::make_shared<v0::Convert>(clamp, element::i8));
        } else {
            output = context.mark_node(std::make_shared<v0::Convert>(quantized_input, element::i32));
        }

        return context.mark_node(std::make_shared<QuantizedPtNode>(quantized_pt_node->get_type(),
                                             context,
                                             output,
                                             scale,
                                             zero_point,
                                             dtype));

    } else if ((quantized_pt_node = cast_quantized_fw_node(input.get_node_shared_ptr(), QuantizedPtNode::quantize_per_channel))) {
        const auto input = quantized_pt_node->get_input_node_ptr(0);
        const auto dtype = quantized_node->get_input_element_type(0);
        const auto scale = quantized_pt_node->get_scale();
        const auto zero_point = quantized_pt_node->get_zero_point();
        const auto axis = quantized_pt_node->get_axis();
        return context.mark_node(std::make_shared<QuantizedPtNode>(quantized_pt_node->get_type(),
                                             context,
                                             input,
                                             scale,
                                             zero_point,
                                             axis,
                                             dtype));
    }
    FRONT_END_OP_CONVERSION_CHECK(false, "Got unknown quantization method in quantize.");
}

std::shared_ptr<ov::Node> dequantize(const NodeContext& context, std::shared_ptr<ov::Node>& quantized_node) {
    std::shared_ptr<QuantizedPtNode> quantized_pt_node; 
    if((quantized_pt_node = cast_quantized_fw_node(quantized_node, QuantizedPtNode::quantize_per_tensor))) {
        const auto input_convert_f32 = context.mark_node(std::make_shared<v0::Convert>(quantized_pt_node->get_input_node_ptr(0), element::f32));
        const auto scale_convert_f32 = context.mark_node(std::make_shared<v0::Convert>(quantized_pt_node->get_scale(), element::f32));
        const auto zero_point_convert_f32 = context.mark_node(std::make_shared<v0::Convert>(quantized_pt_node->get_zero_point(), element::f32));

        const auto input_sub_zero_pt = context.mark_node(std::make_shared<v1::Subtract>(input_convert_f32, zero_point_convert_f32));
        return context.mark_node(std::make_shared<v1::Multiply>(input_sub_zero_pt, scale_convert_f32));
    } else if ((quantized_pt_node = cast_quantized_fw_node(quantized_node, QuantizedPtNode::quantize_per_channel))) {
        // TODO
    }
    FRONT_END_OP_CONVERSION_CHECK(false, "Got unknown quantization method in dequantize.");
}

std::shared_ptr<QuantizedPtNode> cast_quantized_fw_node(std::shared_ptr<Node> node) {
    auto quant_node = std::dynamic_pointer_cast<QuantizedPtNode>(node);
    if (!quant_node) {
        return nullptr;
    }
    const auto& attrs = quant_node->get_attrs();
    if (attrs.find(QuantizedPtNode::quantized_node_type_key) == attrs.end()) {
        return nullptr;
    }
    return quant_node;
}

std::shared_ptr<QuantizedPtNode> cast_quantized_fw_node(std::shared_ptr<Node> node, const std::string& type) {
    auto quant_node = std::dynamic_pointer_cast<QuantizedPtNode>(node);
    if (!quant_node) {
        return nullptr;
    }
    const auto& attrs = quant_node->get_attrs();
    if (attrs.find(QuantizedPtNode::quantized_node_type_key) == attrs.end() ||
        attrs.at(QuantizedPtNode::quantized_node_type_key) != type) {
        return nullptr;
    }
    return quant_node;
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
