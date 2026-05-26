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

    GatedMLP(const ov::Output<Node>& src,
             const ov::Output<Node>& w_gate,
             const ov::Output<Node>& w_up,
             const ov::Output<Node>& w_down,
             const ov::Output<Node>& scale_gate,
             const ov::Output<Node>& scale_up,
             const ov::Output<Node>& scale_down,
             ov::op::internal::GLU::GluType activation,
             const ov::element::Type output_type = ov::element::dynamic);

    GatedMLP(const ov::Output<Node>& src,
             const ov::Output<Node>& w_gate,
             const ov::Output<Node>& w_up,
             const ov::Output<Node>& w_down,
             const ov::Output<Node>& scale_gate,
             const ov::Output<Node>& scale_up,
             const ov::Output<Node>& scale_down,
             const ov::Output<Node>& zp_gate,
             const ov::Output<Node>& zp_up,
             const ov::Output<Node>& zp_down,
             ov::op::internal::GLU::GluType activation,
             const ov::element::Type output_type = ov::element::dynamic);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    ov::op::internal::GLU::GluType get_activation() const { return m_activation; }
    ov::element::Type get_output_type() const { return m_output_type; }
    bool is_compressed_weights() const { return m_compressed_weights; }
    bool has_decompression_zero_points() const { return m_has_decompression_zero_points; }

private:
    ov::op::internal::GLU::GluType m_activation = ov::op::internal::GLU::GluType::Swish;
    ov::element::Type m_output_type = ov::element::dynamic;
    bool m_compressed_weights = false;
    bool m_has_decompression_zero_points = false;
};

}  // namespace ov::intel_gpu::op
