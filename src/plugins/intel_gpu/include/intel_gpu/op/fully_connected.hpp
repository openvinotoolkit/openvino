// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

class FullyConnected : public ov::op::Op {
public:
    OPENVINO_OP("FullyConnected", "gpu_opset");

    FullyConnected() = default;

    FullyConnected(const ov::Output<Node>& A,
                   const ov::Output<Node>& B,
                   const ov::element::Type output_type = ov::element::undefined);

    bool visit_attributes(ov::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    ov::element::Type get_output_type() const { return m_output_type; }

protected:
    ov::element::Type m_output_type;
};

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
