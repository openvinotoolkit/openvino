// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/pytorch/node_context.hpp"
#include "pt_framework_node.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

class QuantizedDecoder : public DummyDecoder {
public:
    QuantizedDecoder(const Output<Node>& input) : m_qinput(input) {}
    virtual PartialShape get_output_shape(size_t index) const override {
        return m_qinput.get_partial_shape();
    }
    virtual const std::string& get_op_type() const override {
        return m_op_type;
    }
    virtual const std::string& get_schema() const override {
        return m_schema;
    }
    virtual size_t num_of_outputs() const override {
        return 1;
    }
    virtual size_t get_subgraph_size() const override {
        return 0;
    }
    virtual const std::string& decoder_type_name() const override {
        return m_decoder_type;
    }

private:
    const Output<Node> m_qinput;
    const std::string m_op_type = "QuantizedPtNode";
    const std::string m_schema = "NONE";
    const std::string m_decoder_type = "qt";
};

enum QuantizedPtNodeType { QUANTIZE_PER_TENSOR, QUANTIZE_PER_CHANNEL };

class QuantizedPtNode : public PtFrameworkNode {
public:
    OPENVINO_OP("QuantizedPtNode", "util", PtFrameworkNode);
    static constexpr const char* quantized_node_type_key = "QuantizedPtTypeName";
    static constexpr const char* quantize_per_tensor = "quantize_per_tensor";
    static constexpr const char* quantize_per_channel = "quantize_per_channel";

    QuantizedPtNode(const QuantizedPtNodeType type,
                    const Output<Node>& input,
                    const Output<Node>& scale,
                    const Output<Node>& zero_point,
                    const element::Type& dtype)
        : PtFrameworkNode(std::make_shared<QuantizedDecoder>(input), {input, scale, zero_point}, 1, false),
          type(type) {
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
                    const Output<Node>& input,
                    const Output<Node>& scale,
                    const Output<Node>& zero_point,
                    const Output<Node>& axis,
                    const element::Type& dtype)
        : PtFrameworkNode(std::make_shared<QuantizedDecoder>(input), {input, scale, zero_point, axis}, 1, false),
          type(type) {
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

    const Output<Node> get_scale() const {
        return input_value(1);
    }
    const Output<Node> get_zero_point() const {
        return input_value(2);
    }
    const Output<Node> get_axis() const {
        if (inputs().size() < 4) {
            return Output<Node>();
        }
        return input_value(3);
    }
    const QuantizedPtNodeType get_type() const {
        return type;
    }
    const element::Type get_dtype() const {
        return dtype;
    }

private:
    const QuantizedPtNodeType type;
    element::Type dtype;
};

/**
 * Quantizes input node with the given parameters. Returns a shared pointer to the new QuantizedPtNode.
 */
Output<Node> quantize(const NodeContext& context,
                      const Output<Node>& input,
                      const Output<Node>& scale,
                      const Output<Node>& zero_point,
                      element::Type dtype,
                      QuantizedPtNodeType quantization_type);
Output<Node> quantize(const NodeContext& context,
                      const Output<Node>& input,
                      const Output<Node>& scale,
                      const Output<Node>& zero_point,
                      const Output<Node>& axis,
                      element::Type dtype,
                      QuantizedPtNodeType quantization_type);

/**
 * Quantizes input node like the quantized node. Returns a shared pointer to the new QuantizedPtNode.
 */
Output<Node> quantize(const NodeContext& context, Output<Node> input, Output<Node> quantized_node);

/**
 * Quantizes input node like the quantized node, with new scale and zero_point parameters. Returns a shared pointer to
 * the new QuantizedPtNode.
 */
Output<Node> quantize(const NodeContext& context,
                      const Output<Node>& input,
                      const Output<Node>& scale,
                      const Output<Node>& zero_point,
                      const Output<Node>& quantized_node);

/**
 * Quantizes input node with the given parameters. Returns a shared pointer to the new QuantizedPtNode.
 */
Output<Node> quantize_fx(const NodeContext& context,
                         const Output<Node>& input,
                         const Output<Node>& scale,
                         const Output<Node>& zero_point,
                         int64_t out_low_i64,
                         int64_t out_high_i64,
                         element::Type dtype,
                         QuantizedPtNodeType quantization_type);
Output<Node> quantize_fx(const NodeContext& context,
                         const Output<Node>& input,
                         const Output<Node>& scale,
                         const Output<Node>& zero_point,
                         const Output<Node>& axis,
                         int64_t out_low_i64,
                         int64_t out_high_i64,
                         element::Type dtype,
                         QuantizedPtNodeType quantization_type);

std::shared_ptr<QuantizedPtNode> cast_quantized_fw_node(std::shared_ptr<Node> node);

namespace op {
/**
 * Modifies conversion function to support quantized case. When input is quantized it is processed as quantized op.
 */
template <OutputVector (*T)(const NodeContext&), size_t in_idx = 0, size_t out_idx = 0>
OutputVector quantizable_op(const NodeContext& context) {
    auto translation_res = T(context);
    FRONT_END_OP_CONVERSION_CHECK(translation_res.size() > out_idx, "Not enough outputs to apply quantization.");
    auto target_input = context.get_input(in_idx);
    if (const auto quantized_pt_node = cast_quantized_fw_node(target_input.get_node_shared_ptr())) {
        return {quantize(context,
                         translation_res[out_idx],
                         quantized_pt_node->get_scale(),
                         quantized_pt_node->get_zero_point(),
                         target_input)};
    }
    return translation_res;
}
}  // namespace op

/**
 * Captures aten::stack([aten::bitwise_and(Constant(u8)), aten::bitwise_right_shift(Constant(u8))], dim=-1).
 * This pattern is transformed to a single Constant with element_type=u4.
 */
std::shared_ptr<Node> u4_compression_stack(const OutputVector& list_elems, int64_t axis);

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
