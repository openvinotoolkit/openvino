// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/pytorch/node_context.hpp"
#include "pt_framework_node.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

enum QuantizedPtNodeType { QUANTIZE_PER_TENSOR, QUANTIZE_PER_CHANNEL, DEQUANTIZE };

class QuantizedPtNode : public PtFrameworkNode {
public:
    OPENVINO_OP("QuantizedPtNode", "util", ::ov::frontend::pytorch::PtFrameworkNode);
    static constexpr const char* quantized_node_type_key = "QuantizedPtTypeName";
    static constexpr const char* quantize_per_tensor_op_type_value = "aten::quantize_per_tensor";
    static constexpr const char* quantize_per_channel_op_type_value = "aten::quantize_per_channel";
    static constexpr const char* dequantize_op_type_value = "aten::dequantize";

    QuantizedPtNode(const QuantizedPtNodeType type,
                    const std::shared_ptr<TorchDecoder>& decoder,
                    const OutputVector& inputs,
                    size_t output_size,
                    bool is_backprop = false)
        : PtFrameworkNode(decoder, inputs, output_size, is_backprop) {
        ov::op::util::FrameworkNodeAttrs attrs = get_attrs();
        if (type == QuantizedPtNodeType::QUANTIZE_PER_TENSOR) {
            attrs[quantized_node_type_key] = quantize_per_tensor_op_type_value;
        } else if (type == QuantizedPtNodeType::QUANTIZE_PER_CHANNEL) {
            attrs[quantized_node_type_key] = quantize_per_channel_op_type_value;
        } else if (type == QuantizedPtNodeType::DEQUANTIZE) {
            attrs[quantized_node_type_key] = dequantize_op_type_value;
        } else {
            FRONT_END_OP_CONVERSION_CHECK(false, "Unknown QuantizedPtNodeType: ", type);
        }
        set_attrs(attrs);
    }
};

std::shared_ptr<QuantizedPtNode> quantize_per_tensor(std::shared_ptr<TorchDecoder> decoder,
                                                     ov::Output<ov::Node> input,
                                                     ov::Output<ov::Node> scale,
                                                     ov::Output<ov::Node> zero_point,
                                                     ov::Output<ov::Node> dtype);

std::shared_ptr<QuantizedPtNode> dequantize(std::shared_ptr<TorchDecoder> decoder, ov::Output<ov::Node> input);

std::shared_ptr<QuantizedPtNode> cast_quantized_fw_node(std::shared_ptr<Node> node);
std::shared_ptr<QuantizedPtNode> cast_quantized_fw_node(std::shared_ptr<Node> node, const std::string& type);

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
