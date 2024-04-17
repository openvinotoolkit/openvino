// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

class MhaFusion : public ov::op::Op {
public:
    OPENVINO_OP("MhaFusion", "gpu_opset");

    MhaFusion() = default;

    /// \brief    Constructs Operation for Multi-Head Attention
    ///
    /// \param input_q         Matrix Q.
    /// \param input_k         Matrix K.
    /// \param input_v         Matrix V.
    MhaFusion(const Output<Node>& input_q,
              const Output<Node>& input_k,
              const Output<Node>& input_v,
              const ov::element::Type output_type = ov::element::undefined);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    ov::element::Type get_output_type() const { return m_output_type; }

protected:
    ov::element::Type m_output_type;
};

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
