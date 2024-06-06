// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace intel_cpu {

class LLMQKVProjNode : public ov::op::Op {
public:
    OPENVINO_OP("LLMQKVProj", "cpu_plugin_opset");

    LLMQKVProjNode() = default;

    struct Config {
        // Fused QKV projection has 3 outputs:
        //   input   :    [M, L, hidden_size]
        // output q_proj: [M, L, hidden_size]
        // output k_proj: [M, L, hidden_size]
        // output v_proj: [M, L, hidden_size]
        int hidden_size;
    };

    // args:
    //      0: input
    //      1: gate_proj
    //      2: up_proj
    //      3: down_proj
    LLMQKVProjNode(const OutputVector& args, const Config& cfg) : Op(args), m_config(cfg) {
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
}  // namespace ov
