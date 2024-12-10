// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/op/util/variable_extension.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

/// \brief Similar to common v6::ReadValue, but it's not derived from ReadValueBase class to avoid ReadValue-Assign pairing check
/// This is needed to have ReadValue-KVCache pair instead of ReadValue-Assign
class ReadValue : public ov::op::Op, public ov::op::util::VariableExtension {
public:
    OPENVINO_OP("ReadValue", "gpu_opset");

    ReadValue() = default;

    ReadValue(const std::shared_ptr<ov::op::util::Variable>& variable);
    ReadValue(const Output<Node>& variable_initializer, const std::shared_ptr<ov::op::util::Variable>& variable);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;
    void validate_and_infer_types(size_t output_idx, const ov::op::util::VariableInfo& variable_info);

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    std::string get_variable_id() const override {
        OPENVINO_ASSERT(m_variable, "Variable is not initialized. Variable_id is unavailable");
        return m_variable->get_info().variable_id;
    }

protected:
    ReadValue(const std::vector<Output<Node>>& variable_initializers, const std::shared_ptr<ov::op::util::Variable>& variable)
    : Op(variable_initializers) {
        m_variable = variable;
    }
};

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
