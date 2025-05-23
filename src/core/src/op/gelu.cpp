// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gelu.hpp"

#include <cmath>

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/core/type.hpp"
#include "openvino/reference/gelu.hpp"

namespace ov {
namespace op {
namespace v0 {
Gelu::Gelu() : UnaryElementwiseArithmetic() {}

Gelu::Gelu(const Output<Node>& data) : UnaryElementwiseArithmetic(data) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> Gelu::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Gelu_clone_with_new_inputs);
    if (new_args.size() != 1) {
        OPENVINO_THROW("Incorrect number of new arguments");
    }
    return std::make_shared<Gelu>(new_args.at(0));
}

void Gelu::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Gelu_validate_and_infer_types);
    element::Type input_element_type = get_input_element_type(0);
    PartialShape input_pshape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");

    set_output_type(0, input_element_type, input_pshape);
}
}  // namespace v0

namespace v7 {
Gelu::Gelu(const Output<Node>& data, GeluApproximationMode mode)
    : UnaryElementwiseArithmetic(data),
      m_approximation_mode(mode) {
    constructor_validate_and_infer_types();
}

bool Gelu::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v7_Gelu_visit_attributes);
    visitor.on_attribute("approximation_mode", m_approximation_mode);
    return true;
}

std::shared_ptr<Node> Gelu::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v7_Gelu_clone_with_new_inputs);
    if (new_args.size() != 1) {
        OPENVINO_THROW("Incorrect number of new arguments");
    }
    return std::make_shared<Gelu>(new_args.at(0), m_approximation_mode);
}

void Gelu::validate_and_infer_types() {
    OV_OP_SCOPE(v7_Gelu_validate_and_infer_types);
    element::Type input_element_type = get_input_element_type(0);
    PartialShape input_pshape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");

    set_output_type(0, input_element_type, input_pshape);
}

op::GeluApproximationMode Gelu::get_approximation_mode() const {
    return m_approximation_mode;
}

namespace gelu {
namespace {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& in, Tensor& out, const op::GeluApproximationMode mode, const size_t count) {
        reference::gelu(in.data<const T>(), out.data<T>(), mode, count);
        return true;
    }
};
}  // namespace
}  // namespace gelu

bool Gelu::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v7_Gelu_evaluate);
    OPENVINO_ASSERT(inputs.size() == 1 && outputs.size() == 1);

    const auto& input_shape = inputs[0].get_shape();
    const auto count = shape_size(input_shape);
    outputs[0].set_shape(input_shape);
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v7_Gelu_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32),
                                      gelu::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      m_approximation_mode,
                                      count);
}

bool Gelu::has_evaluate() const {
    OV_OP_SCOPE(v7_Gelu_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::f16:
    case element::f32:
        return true;
    default:
        return false;
    }
}
}  // namespace v7
}  // namespace op

template <>
OPENVINO_API EnumNames<op::GeluApproximationMode>& EnumNames<op::GeluApproximationMode>::get() {
    static auto enum_names = EnumNames<op::GeluApproximationMode>(
        "op::GeluApproximationMode",
        {{"TANH", op::GeluApproximationMode::TANH}, {"ERF", op::GeluApproximationMode::ERF}});
    return enum_names;
}

std::ostream& op::operator<<(std::ostream& s, const op::GeluApproximationMode& type) {
    return s << as_string(type);
}

AttributeAdapter<op::GeluApproximationMode>::~AttributeAdapter() = default;
}  // namespace ov
