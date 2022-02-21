// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>

namespace ov {
namespace intel_cpu {

class LeakyReluNode : public ngraph::op::Op {
public:
    OPENVINO_OP("LeakyRelu", "cpu_plugin_opset");

    LeakyReluNode() = default;

    LeakyReluNode(const ngraph::Output<ngraph::Node> &data, const float &negative_slope, const ngraph::element::Type output_type);

    void validate_and_infer_types() override;

    bool visit_attributes(ngraph::AttributeVisitor &visitor) override;

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector &new_args) const override;

    float get_slope() { return m_negative_slope; }

    ngraph::element::Type get_output_type() const { return m_output_type; }

private:
    float m_negative_slope = 0.f;
    ngraph::element::Type m_output_type;
};

}   // namespace intel_cpu
}   // namespace ov
