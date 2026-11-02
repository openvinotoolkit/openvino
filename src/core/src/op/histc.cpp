// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/histc.hpp"

#include <cmath>

#include "histc_shape_inference.hpp"
#include "itt.hpp"

namespace ov::op::v17 {

Histc::Histc(const Output<Node>& data,
             const int64_t bins,
             const double min,
             const double max,
             const OutputType output_type)
    : Op({data}),
      m_bins{bins},
      m_min{min},
      m_max{max},
      m_output_type{output_type} {
    constructor_validate_and_infer_types();
}

bool Histc::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v17_Histc_visit_attributes);
    visitor.on_attribute("bins", m_bins);
    visitor.on_attribute("min", m_min);
    visitor.on_attribute("max", m_max);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void Histc::validate_and_infer_types() {
    OV_OP_SCOPE(v17_Histc_validate_and_infer_types);
    const auto& data_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          data_et.is_dynamic() || data_et.is_real() || data_et.is_integral_number(),
                          "Input element type must be numeric. Got: ",
                          data_et);
    NODE_VALIDATION_CHECK(this,
                          std::isfinite(m_min) && std::isfinite(m_max),
                          "Attributes 'min' and 'max' must be finite. Got: ",
                          m_min,
                          " and ",
                          m_max);
    NODE_VALIDATION_CHECK(this,
                          m_min <= m_max,
                          "Attribute 'min' must be less or equal than 'max'. Got: ",
                          m_min,
                          " and ",
                          m_max);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, m_output_type == OutputType::I32 ? element::i32 : data_et, output_shapes[0]);
}

std::shared_ptr<Node> Histc::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v17_Histc_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Histc>(new_args.at(0), m_bins, m_min, m_max, m_output_type);
}
}  // namespace ov::op::v17

namespace ov {
std::ostream& operator<<(std::ostream& s, const op::v17::Histc::OutputType& type) {
    return s << as_string(type);
}

template <>
OPENVINO_API EnumNames<op::v17::Histc::OutputType>& EnumNames<op::v17::Histc::OutputType>::get() {
    static auto enum_names = EnumNames<op::v17::Histc::OutputType>(
        "op::v17::Histc::OutputType",
        {{"same_as_input", op::v17::Histc::OutputType::SAME_AS_INPUT}, {"i32", op::v17::Histc::OutputType::I32}});
    return enum_names;
}

AttributeAdapter<op::v17::Histc::OutputType>::~AttributeAdapter() = default;
}  // namespace ov
