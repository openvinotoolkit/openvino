// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/rms_norm.hpp"

#include "itt.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v14 {

RMSNorm::RMSNorm(const Output<Node>& data,
                 const Output<Node>& scale,
                 double epsilson,
                 const ov::element::Type& compute_type)
    : Op({data, scale}),
      m_epsilon(epsilson),
      m_compute_type(compute_type) {
    constructor_validate_and_infer_types();
}

RMSNorm::RMSNorm(const Output<Node>& data, double epsilson, const ov::element::Type& compute_type)
    : Op({data}),
      m_epsilon(epsilson),
      m_compute_type(compute_type) {
    constructor_validate_and_infer_types();
}

bool RMSNorm::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(v14_RMSNorm_visit_attributes);
    visitor.on_attribute("epsilon", m_epsilon);
    visitor.on_attribute("compute_type", m_compute_type);
    return true;
}

void RMSNorm::validate_and_infer_types() {
    OV_OP_SCOPE(v14_RMSNorm_validate_and_infer_types);
    const auto& data_element_type = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          data_element_type.is_dynamic() || data_element_type.is_real(),
                          "The element type of the input tensor must be a floating point type.");
    set_output_type(0, data_element_type, get_input_partial_shape(0));
}

std::shared_ptr<Node> RMSNorm::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(v14_RMSNorm_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 1) {
        return std::make_shared<RMSNorm>(new_args.at(0), m_epsilon, m_compute_type);
    }
    return std::make_shared<RMSNorm>(new_args.at(0), new_args.at(1), m_epsilon, m_compute_type);
}

}  // namespace v14
}  // namespace op
}  // namespace ov
