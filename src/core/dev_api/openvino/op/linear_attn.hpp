// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/op/util/variable_extension.hpp"

namespace ov {
namespace op {

// This is an experimental operation that is implemented in the plugins.
// Do not use in user applications, backward compatibility is not guaranteed in future releases.
class OPENVINO_API LinearAttention : public ov::op::Op, public ov::op::util::VariableExtension {
public:
    OPENVINO_OP("LinearAttention");

    LinearAttention() = default;

    LinearAttention(const ov::OutputVector& args);
    LinearAttention(const ov::OutputVector& args, const std::shared_ptr<ov::op::util::Variable>& variable);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    std::string get_variable_id() const override {
        OPENVINO_ASSERT(m_variable, "Variable is not initialized. Variable_id is unavailable");
        return m_variable->get_info().variable_id;
    }

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    void set_out_type(int index, const ov::element::Type& output_type);

protected:
    std::vector<ov::element::Type> m_output_type = {ov::element::dynamic, ov::element::dynamic, ov::element::dynamic};
};

}  // namespace op
}  // namespace ov
