// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/op/read_value.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

/// \brief This operation handles the OpenVINO GPU Plugin's custom variable
//         representation (which can store multiple states in a single variable) at the graph level.
class ReadValues : public ReadValue {
public:
    OPENVINO_OP("ReadValues", "gpu_opset");

    ReadValues() = default;

    ReadValues(const std::shared_ptr<ov::op::util::Variable>& variable,
               const std::vector<ov::op::util::VariableInfo>& internal_states_infos);

    ReadValues(const OutputVector& variable_initializers,
               const std::shared_ptr<ov::op::util::Variable>& variable,
               const std::vector<ov::op::util::VariableInfo>& internal_states_infos);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::vector<ov::op::util::VariableInfo> get_all_internal_states_info() const;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

private:
    std::vector<ov::op::util::VariableInfo> m_internal_states_infos;
};

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
