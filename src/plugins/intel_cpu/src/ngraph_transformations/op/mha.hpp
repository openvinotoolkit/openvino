// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>

namespace ov {
namespace intel_cpu {

class MHANode : public ngraph::op::Op {
public:
    OPENVINO_OP("MHA", "cpu_plugin_opset");

    MHANode() = default;

    MHANode(const ngraph::Output<ngraph::Node> &in0,
            const ngraph::Output<ngraph::Node> &in1,
            const ngraph::Output<ngraph::Node> &in2,
            const ngraph::Output<ngraph::Node> &in3,
            const ngraph::Output<ngraph::Node> &in4,
            const ngraph::element::Type output_type);

    MHANode(const ngraph::Output<ngraph::Node> &in0,
            const ngraph::Output<ngraph::Node> &in1,
            const ngraph::Output<ngraph::Node> &in2,
            const ngraph::Output<ngraph::Node> &in3,
            const ngraph::Output<ngraph::Node> &in4,
            const std::vector<float> &fq_scales0,
            const std::vector<float> &fq_scales1,
            const std::vector<float> &fq_scales2,
            const ngraph::element::Type output_type);


    void validate_and_infer_types() override;

    bool visit_attributes(ngraph::AttributeVisitor &visitor) override;

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector &new_args) const override;

    ngraph::element::Type get_output_type() const { return m_output_type; }

    bool with_fq_scales() const {
        return !fq_scales.empty();
    }

    const std::vector<float>& get_fq_scales(size_t idx) const {
        NODE_VALIDATION_CHECK(this,
                        fq_scales.size() > idx,
                        "Accessing quantization scales by invalid index: ",
                        idx);

        return fq_scales[idx];
    }

private:
    ngraph::element::Type m_output_type;
    std::vector<std::vector<float>> fq_scales;
};

}   // namespace intel_cpu
}   // namespace ov
