// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/histc.hpp"

#include <cmath>

#include "element_visitor.hpp"
#include "histc_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/reference/histc.hpp"

namespace ov::op {
namespace histc {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& in, Tensor& out, const size_t bins, const double min, const double max) {
        reference::histc(in.data<const T>(), out.data<T>(), in.get_size(), bins, min, max);
        return true;
    }
};
}  // namespace histc

namespace v17 {

Histc::Histc(const Output<Node>& data, const int64_t bins, const double min, const double max)
    : Op({data}),
      m_bins{bins},
      m_min{min},
      m_max{max} {
    constructor_validate_and_infer_types();
}

bool Histc::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v17_Histc_visit_attributes);
    visitor.on_attribute("bins", m_bins);
    visitor.on_attribute("min", m_min);
    visitor.on_attribute("max", m_max);
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
    set_output_type(0, data_et, output_shapes[0]);
}

std::shared_ptr<Node> Histc::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v17_Histc_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Histc>(new_args.at(0), m_bins, m_min, m_max);
}

bool Histc::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v17_Histc_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 1);

    outputs[0].set_shape(Shape{static_cast<size_t>(m_bins)});

    using namespace ov::element;
    return IF_TYPE_OF(v17_Histc_evaluate,
                      OV_PP_ET_LIST(f16, bf16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64),
                      histc::Evaluate,
                      inputs[0].get_element_type(),
                      inputs[0],
                      outputs[0],
                      static_cast<size_t>(m_bins),
                      m_min,
                      m_max);
}

bool Histc::has_evaluate() const {
    OV_OP_SCOPE(v17_Histc_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::f16:
    case element::bf16:
    case element::f32:
    case element::f64:
    case element::i8:
    case element::i16:
    case element::i32:
    case element::i64:
    case element::u8:
    case element::u16:
    case element::u32:
    case element::u64:
        return true;
    default:
        return false;
    }
}
}  // namespace v17
}  // namespace ov::op
