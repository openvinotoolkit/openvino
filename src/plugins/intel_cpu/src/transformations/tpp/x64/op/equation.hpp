// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "modifiers.hpp"
#include "openvino/op/op.hpp"
#include "descriptor.hpp"

namespace ov {
namespace intel_cpu {
namespace tpp {
namespace op {

class EquationTPP : public modifier::TensorProcessingPrimitive, public ov::op::Op {
public:
    OPENVINO_OP("EquationTPP", "TppOpset", ov::op::Op);
    EquationTPP(const OutputVector& arguments, std::vector<OpDescTPP> op_descs);
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    const std::vector<OpDescTPP>& get_op_descs() { return m_op_descs; }

private:
    std::vector<OpDescTPP> m_op_descs;
};

} // namespace op
} // namespace tpp
} // namespace intel_cpu
} // namespace ov
