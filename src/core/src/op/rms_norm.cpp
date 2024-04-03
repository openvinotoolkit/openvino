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
    const bool is_valid_data_type = data_element_type.is_dynamic() || data_element_type.is_real();
    NODE_VALIDATION_CHECK(this,
                          is_valid_data_type,
                          "The element type of the data tensor must be a floating point type. Got: ",
                          data_element_type);

    const auto& axes_element_type = get_input_element_type(1);
    const bool is_valid_axes_type =
        data_element_type.is_dynamic() || axes_element_type == element::i32 || axes_element_type == element::i64;
    NODE_VALIDATION_CHECK(this,
                          is_valid_axes_type,
                          "The element type of the axes tensor must be i32 or i64 type. Got: ",
                          axes_element_type);

    const auto& data_shape = get_input_partial_shape(0);
    const auto& axes_shape = get_input_partial_shape(1);
    if (axes_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(this,
                              axes_shape.size() == 1,
                              "Expected 1D tensor for the 'axes' input. Got: ",
                              axes_shape);

        const auto data_rank = data_shape.rank();
        const bool has_axes_compatible = data_rank.is_dynamic() || axes_shape[0].is_dynamic() ||
                                         cmp::ge(data_rank.get_length(), axes_shape.get_shape()[0]);
        NODE_VALIDATION_CHECK(this,
                              has_axes_compatible,
                              "Number of the axes can't be higher than the rank of the data shape.");
    }

    if (get_input_size() > 2) {  // Validate scale input
        auto scale_shape = get_input_partial_shape(2);
        const bool is_scale_shape_broadcastable =
            PartialShape::broadcast_merge_into(scale_shape, data_shape, ov::op::AutoBroadcastType::NUMPY);
        NODE_VALIDATION_CHECK(this,
                              is_scale_shape_broadcastable,
                              "Scale input shape must be broadcastable to the shape of the data input.");
    }

    // Output type and shape is the same as the first input
    set_output_type(0, data_element_type, get_input_partial_shape(0));
}

std::shared_ptr<Node> RMSNorm::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(v14_RMSNorm_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 1) {
        return std::make_shared<RMSNorm>(new_args.at(0), new_args.at(1), m_epsilon, m_compute_type);
    }
    return std::make_shared<RMSNorm>(new_args.at(0), new_args.at(1), new_args.at(2), m_epsilon, m_compute_type);
}

}  // namespace v14
}  // namespace op
}  // namespace ov
