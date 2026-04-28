// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scaled_shifted_clamp_experimental.hpp"

#include "itt.hpp"

namespace ov {
namespace op {
namespace experimental {

ScaledShiftedClamp::ScaledShiftedClamp(const Output<Node>& data, double scale, double bias, double lo, double hi)
    : Op({data}),
      m_scale(scale),
      m_bias(bias),
      m_lo(lo),
      m_hi(hi) {
    constructor_validate_and_infer_types();
}

bool ScaledShiftedClamp::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(experimental_ScaledShiftedClamp_visit_attributes);
    visitor.on_attribute("scale", m_scale);
    visitor.on_attribute("bias", m_bias);
    visitor.on_attribute("lo", m_lo);
    visitor.on_attribute("hi", m_hi);
    return true;
}

void ScaledShiftedClamp::validate_and_infer_types() {
    OV_OP_SCOPE(experimental_ScaledShiftedClamp_validate_and_infer_types);

    const auto& data_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          data_et.is_dynamic() || data_et.is_real(),
                          "Input element type must be a floating point type. Got: ",
                          data_et);
    NODE_VALIDATION_CHECK(this, m_lo <= m_hi, "Attribute `lo` must be <= `hi`. Got lo=", m_lo, ", hi=", m_hi);

    set_output_type(0, data_et, get_input_partial_shape(0));
}

std::shared_ptr<Node> ScaledShiftedClamp::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(experimental_ScaledShiftedClamp_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ScaledShiftedClamp>(new_args.at(0), m_scale, m_bias, m_lo, m_hi);
}

}  // namespace experimental
}  // namespace op
}  // namespace ov
