// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>

namespace ov {
namespace intel_cpu {

class PowerStaticNode : public ngraph::op::Op {
public:
    OPENVINO_OP("PowerStatic", "cpu_plugin_opset");

    PowerStaticNode() = default;

    PowerStaticNode(const ngraph::Output<ngraph::Node> &data, const float &power, const float &scale, const float &shift,
                    const ngraph::element::Type output_type = ngraph::element::undefined);

    void validate_and_infer_types() override;

    bool visit_attributes(ngraph::AttributeVisitor &visitor) override;

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector &new_args) const override;

    float get_power() const { return power; }
    float get_scale() const { return scale; }
    float get_shift() const { return shift; }

private:
    float scale, power, shift;
    ngraph::element::Type m_output_type;
};

}   // namespace intel_cpu
}   // namespace ov
