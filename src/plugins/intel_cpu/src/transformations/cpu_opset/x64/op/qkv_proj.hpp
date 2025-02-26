// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov::intel_cpu {

class QKVProjectionNode : public ov::op::Op {
public:
    OPENVINO_OP("QKVProjection", "cpu_plugin_opset");

    QKVProjectionNode() = default;

    struct Config {
        bool quantized;
        int hidden_size;
        int proj_size0;
        int proj_size1;
        int proj_size2;
        bool weights_combined;
    };

    QKVProjectionNode(const OutputVector& args, const Config& cfg) : Op(args), m_config(cfg) {
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

}  // namespace ov::intel_cpu
