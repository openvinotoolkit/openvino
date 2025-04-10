// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::intel_cpu {

class MHANode : public ov::op::Op {
public:
    OPENVINO_OP("MHA", "cpu_plugin_opset");

    MHANode() = default;

    MHANode(const ov::Output<ov::Node>& in0,
            const ov::Output<ov::Node>& in1,
            const ov::Output<ov::Node>& in2,
            const ov::Output<ov::Node>& in3,
            std::vector<float> mul_scales,
            bool is_mul_first,
            const ov::element::Type output_type);

    MHANode(const ov::Output<ov::Node>& in0,
            const ov::Output<ov::Node>& in1,
            const ov::Output<ov::Node>& in2,
            const ov::Output<ov::Node>& in3,
            std::vector<float> mul_scales,
            bool is_mul_first,
            std::vector<float> fq_scales0,
            std::vector<float> fq_scales1,
            std::vector<float> fq_scales2,
            std::vector<float> fq_scales3,
            const ov::element::Type fq0_output_type,
            const ov::element::Type fq1_output_type,
            const ov::element::Type fq2_output_type,
            const ov::element::Type output_type);

    void validate_and_infer_types() override;

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    ov::element::Type get_output_type() const {
        return m_output_type;
    }

    const std::vector<float>& get_mul_scales() const {
        return mul_scales;
    }

    const std::vector<float>& get_fq_scales0() const {
        return fq_scales0;
    }
    const std::vector<float>& get_fq_scales1() const {
        return fq_scales1;
    }
    const std::vector<float>& get_fq_scales2() const {
        return fq_scales2;
    }
    const std::vector<float>& get_fq_scales3() const {
        return fq_scales3;
    }

    bool get_is_mul_first() const {
        return is_mul_first;
    }

    ov::element::Type get_fq0_output_type() const {
        return fq0_output_type;
    }
    ov::element::Type get_fq1_output_type() const {
        return fq1_output_type;
    }
    ov::element::Type get_fq2_output_type() const {
        return fq2_output_type;
    }

private:
    ov::element::Type m_output_type;
    std::vector<float> mul_scales;
    bool is_mul_first;
    std::vector<float> fq_scales0;
    std::vector<float> fq_scales1;
    std::vector<float> fq_scales2;
    std::vector<float> fq_scales3;
    ov::element::Type fq0_output_type;
    ov::element::Type fq1_output_type;
    ov::element::Type fq2_output_type;
};

}  // namespace ov::intel_cpu
