// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"
#include "ov_ops/glu.hpp"

namespace ov::intel_gpu::op {

class GatedMLP : public ov::op::Op {
public:
    OPENVINO_OP("GatedMLP", "gpu_opset");

    GatedMLP() = default;

    GatedMLP(const ov::Output<Node>& src,
             const ov::Output<Node>& w_gate,
             const ov::Output<Node>& w_up,
             const ov::Output<Node>& w_down,
             ov::op::internal::GLU::GluType activation,
             const ov::element::Type output_type = ov::element::dynamic);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    ov::op::internal::GLU::GluType get_activation() const { return m_activation; }
    ov::element::Type get_output_type() const { return m_output_type; }

private:
    ov::op::internal::GLU::GluType m_activation = ov::op::internal::GLU::GluType::Swish;
    ov::element::Type m_output_type = ov::element::dynamic;
};

}  // namespace ov::intel_gpu::op
