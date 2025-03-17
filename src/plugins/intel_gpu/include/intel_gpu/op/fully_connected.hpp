// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov::intel_gpu::op {

class FullyConnected : public ov::op::Op {
public:
    OPENVINO_OP("FullyConnected", "gpu_opset");

    FullyConnected() = default;

    FullyConnected(const ov::Output<Node>& A,
                   const ov::Output<Node>& B,
                   const ov::Output<Node>& bias,
                   const ov::element::Type output_type = ov::element::dynamic);

    bool visit_attributes(ov::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    ov::element::Type get_output_type() const { return m_output_type; }

protected:
    ov::element::Type m_output_type;
};

}   // namespace ov::intel_gpu::op
