// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/rms_norm.hpp"

#include "compare.hpp"
#include "itt.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v14 {

RMSNorm::RMSNorm(const Output<Node>& data,
                 const Output<Node>& axes,
                 double epsilson,
                 const ov::element::Type& compute_type)
    : Op({data, axes}),
      m_epsilon(epsilson),
      m_compute_type(compute_type) {
    constructor_validate_and_infer_types();
}

RMSNorm::RMSNorm(const Output<Node>& data,
                 const Output<Node>& axes,
                 const Output<Node>& scale,
                 double epsilson,
                 const ov::element::Type& compute_type)
    : Op({data, axes, scale}),
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
    const auto& data = get_input_partial_shape(0);
    const auto& axes = get_input_partial_shape(1);

    if (axes.is_static()) {
        NODE_VALIDATION_CHECK(this, is_vector(axes.to_shape()), "Expected 1D tensor for the 'axes' input. Got: ", axes);

        const auto data_rank = data.rank();
        NODE_VALIDATION_CHECK(this,
                              data_rank.is_dynamic() || cmp::ge(data_rank.get_length(), axes.get_shape()[0]),
                              "Expected rank for the 'data' input to be higher than axes shape. Got: ",
                              data);
    }
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
