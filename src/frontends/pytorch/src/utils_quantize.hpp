// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/pytorch/node_context.hpp"
#include "pt_framework_node.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

enum QuantizedPtNodeType { QUANTIZE_PER_TENSOR, QUANTIZE_PER_CHANNEL };

class QuantizedPtNode : public PtFrameworkNode {
public:
    OPENVINO_OP("QuantizedPtNode", "util", ::ov::frontend::pytorch::PtFrameworkNode);
    static constexpr const char* quantized_node_type_key = "QuantizedPtTypeName";
    static constexpr const char* quantize_per_tensor = "quantize_per_tensor";
    static constexpr const char* quantize_per_channel = "quantize_per_channel";

    QuantizedPtNode(const QuantizedPtNodeType type,
                    const NodeContext& context,
                    const ov::Output<ov::Node> input,
                    const ov::Output<ov::Node> scale,
                    const ov::Output<ov::Node> zero_point)
        : PtFrameworkNode(context.get_decoder(), {input}, 1, false),
          type(type),
          scale(scale),
          zero_point(zero_point) {
        ov::op::util::FrameworkNodeAttrs attrs = get_attrs();
        if (type == QuantizedPtNodeType::QUANTIZE_PER_TENSOR) {
            attrs[quantized_node_type_key] = quantize_per_tensor;
        } else if (type == QuantizedPtNodeType::QUANTIZE_PER_CHANNEL) {
            attrs[quantized_node_type_key] = quantize_per_channel;
        } else {
            FRONT_END_OP_CONVERSION_CHECK(false, "Unknown QuantizedPtNodeType: ", type);
        }
        set_attrs(attrs);
    }

    QuantizedPtNode(const QuantizedPtNodeType type,
                    const NodeContext& context,
                    const ov::Output<ov::Node>& input,
                    const ov::Output<ov::Node>& scale,
                    const ov::Output<ov::Node>& zero_point,
                    const ov::Output<ov::Node>& axis_)
        : QuantizedPtNode(type, context, input, scale, zero_point) {
        axis = axis_;
    }

    const ov::Output<ov::Node> get_scale() {
        return scale;
    }
    const ov::Output<ov::Node> get_zero_point() {
        return zero_point;
    }
    const ov::Output<ov::Node> get_axis() {
        FRONT_END_OP_CONVERSION_CHECK(
            type == QuantizedPtNodeType::QUANTIZE_PER_CHANNEL,
            "Accessing axis from a node quantized using a method other than 'per channel' is disallowed.");
        return axis;
    }
    const QuantizedPtNodeType get_type() {
        return type;
    }

private:
    const QuantizedPtNodeType type;
    ov::Output<ov::Node> scale;
    ov::Output<ov::Node> zero_point;
    ov::Output<ov::Node> axis;
};

std::shared_ptr<ov::Node> quantize(const NodeContext& context,
                                   std::shared_ptr<ov::Node>& input,
                                   std::shared_ptr<ov::Node>& quantized_node);

std::shared_ptr<ov::Node> dequantize(const NodeContext& context, std::shared_ptr<ov::Node>& quantized_node);

std::shared_ptr<QuantizedPtNode> cast_quantized_fw_node(std::shared_ptr<Node> node);
std::shared_ptr<QuantizedPtNode> cast_quantized_fw_node(std::shared_ptr<Node> node, const std::string& type);

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
