// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace intel_cpu {

class LLMMLPNode : public ov::op::Op {
public:
    OPENVINO_OP("LLMMLP", "cpu_plugin_opset");

    LLMMLPNode() = default;

    enum class ACT_FN { SILU = 0, GELU = 1};

    struct Config {
        ACT_FN act;
        bool gate_up_quantized;
        bool down_quantized;
        int hidden_size;
        int up_size;
        bool gate_up_combined;
    };

    // args:
    //      0: input
    //      1: gate_proj
    //      2: up_proj
    //      3: down_proj
    LLMMLPNode(const OutputVector& args, const Config& cfg) : Op(args), m_config(cfg) {
        validate_and_infer_types();
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const Config& get_config() const {
        return m_config;
    }

private:
    Config m_config;
};

}  // namespace intel_cpu

template <>
class AttributeAdapter<ov::intel_cpu::LLMMLPNode::ACT_FN>
    : public EnumAttributeAdapterBase<ov::intel_cpu::LLMMLPNode::ACT_FN> {
public:
    AttributeAdapter(ov::intel_cpu::LLMMLPNode::ACT_FN& value)
        : EnumAttributeAdapterBase<ov::intel_cpu::LLMMLPNode::ACT_FN>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::intel_cpu::LLMMLPNode::ACT_FN>");
};

std::ostream& operator<<(std::ostream& s, const ov::intel_cpu::LLMMLPNode::ACT_FN& type);

}  // namespace ov
