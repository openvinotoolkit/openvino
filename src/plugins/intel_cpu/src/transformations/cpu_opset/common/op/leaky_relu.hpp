// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::intel_cpu {

class LeakyReluNode : public ov::op::Op {
public:
    OPENVINO_OP("LeakyRelu", "cpu_plugin_opset");

    LeakyReluNode() = default;

    LeakyReluNode(const ov::Output<ov::Node>& data, const float& negative_slope, const ov::element::Type output_type);

    void validate_and_infer_types() override;

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    float get_slope() {
        return m_negative_slope;
    }

    ov::element::Type get_output_type() const {
        return m_output_type;
    }

private:
    float m_negative_slope = 0.f;
    ov::element::Type m_output_type;
};

}  // namespace ov::intel_cpu
