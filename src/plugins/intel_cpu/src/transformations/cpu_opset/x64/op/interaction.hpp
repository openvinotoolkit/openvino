// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov::intel_cpu {

class InteractionNode : public ov::op::Op {
public:
    OPENVINO_OP("Interaction", "cpu_plugin_opset");

    InteractionNode() = default;

    InteractionNode(const OutputVector& args);

    InteractionNode(const NodeVector& args);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    ov::element::Type get_output_type() const {
        return m_output_type;
    }

    void set_fq_scales(const std::vector<float>& scales) {
        m_fq_scales = scales;
    }

    const std::vector<float>& get_output_scales() const {
        return m_fq_scales;
    }

private:
    ov::element::Type m_output_type;
    std::vector<float> m_fq_scales;
};

}  // namespace ov::intel_cpu
