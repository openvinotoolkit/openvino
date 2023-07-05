// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils_quantize.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

std::shared_ptr<QuantizedPtNode> quantize(const NodeContext& context, QuantizedPtNodeType type) {
    FRONT_END_OP_CONVERSION_CHECK(type != QuantizedPtNodeType::DEQUANTIZE, "Quantize called with DEQUANTIZE type");
    return std::make_shared<QuantizedPtNode>(type,
                                             context.get_decoder(),
                                             context.inputs(),
                                             context.get_decoder()->num_of_outputs(),
                                             false);
}

std::shared_ptr<QuantizedPtNode> dequantize(const NodeContext& context) {
    return std::make_shared<QuantizedPtNode>(QuantizedPtNodeType::DEQUANTIZE,
                                             context.get_decoder(),
                                             context.inputs(),
                                             context.get_decoder()->num_of_outputs(),
                                             false);
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
