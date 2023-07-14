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
                    const ov::Output<ov::Node> zero_point,
                    element::Type& dtype)
        : PtFrameworkNode(context.get_decoder(), {input}, 1, false),
          type(type),
          scale(scale.get_node_shared_ptr()),
          zero_point(zero_point.get_node_shared_ptr()),
          axis(nullptr) {
        ov::op::util::FrameworkNodeAttrs attrs = get_attrs();
        if (type == QuantizedPtNodeType::QUANTIZE_PER_TENSOR) {
            attrs[quantized_node_type_key] = quantize_per_tensor;
        } else if (type == QuantizedPtNodeType::QUANTIZE_PER_CHANNEL) {
            FRONT_END_OP_CONVERSION_CHECK(false, "quantize_per_channel requires axis to be provided.");
        } else {
            FRONT_END_OP_CONVERSION_CHECK(false, "Unknown QuantizedPtNodeType: ", type);
        }
        set_attrs(attrs);
        this->dtype = dtype;
    }

    QuantizedPtNode(const QuantizedPtNodeType type,
                    const NodeContext& context,
                    const ov::Output<ov::Node> input,
                    const ov::Output<ov::Node> scale,
                    const ov::Output<ov::Node> zero_point,
                    const ov::Output<ov::Node> axis,
                    element::Type& dtype)
        : PtFrameworkNode(context.get_decoder(), {input}, 1, false),
          type(type),
          scale(scale.get_node_shared_ptr()),
          zero_point(zero_point.get_node_shared_ptr()),
          axis(axis.get_node_shared_ptr()) {
        ov::op::util::FrameworkNodeAttrs attrs = get_attrs();
        if (type == QuantizedPtNodeType::QUANTIZE_PER_TENSOR) {
            attrs[quantized_node_type_key] = quantize_per_tensor;
        } else if (type == QuantizedPtNodeType::QUANTIZE_PER_CHANNEL) {
            attrs[quantized_node_type_key] = quantize_per_channel;
        } else {
            FRONT_END_OP_CONVERSION_CHECK(false, "Unknown QuantizedPtNodeType: ", type);
        }
        set_attrs(attrs);
        this->dtype = dtype;
    }

    const std::shared_ptr<ov::Node> get_scale() {
        return scale;
    }
    const std::shared_ptr<ov::Node> get_zero_point() {
        return zero_point;
    }
    const std::shared_ptr<ov::Node> get_axis() {
        return axis;
    }
    const QuantizedPtNodeType get_type() {
        return type;
    }
    const element::Type get_dtype() {
        return dtype;
    }

private:
    const QuantizedPtNodeType type;
    std::shared_ptr<ov::Node> scale;
    std::shared_ptr<ov::Node> zero_point;
    std::shared_ptr<ov::Node> axis;
    element::Type dtype;
};

/**
 * Quantizes input node with the given parameters. Returns a shared pointer to the new QuantizedPtNode.
 */
ov::Output<ov::Node> quantize(const NodeContext& context,
                              ov::Output<ov::Node> input,
                              ov::Output<ov::Node> scale,
                              ov::Output<ov::Node> zero_point,
                              ov::element::Type dtype,
                              QuantizedPtNodeType quantization_type);
ov::Output<ov::Node> quantize(const NodeContext& context,
                              ov::Output<ov::Node> input,
                              ov::Output<ov::Node> scale,
                              ov::Output<ov::Node> zero_point,
                              ov::Output<ov::Node> axis,
                              ov::element::Type dtype,
                              QuantizedPtNodeType quantization_type);

std::shared_ptr<QuantizedPtNode> cast_quantized_fw_node(ov::Output<ov::Node> node);
std::shared_ptr<QuantizedPtNode> cast_quantized_fw_node(ov::Output<ov::Node> node, const std::string& type);
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
