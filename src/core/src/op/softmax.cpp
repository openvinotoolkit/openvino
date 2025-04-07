// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/softmax.hpp"

#include <algorithm>

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/softmax.hpp"

namespace ov {
namespace op {
namespace softmax {
namespace {
struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& in, Tensor& out, const Shape& shape, const AxisSet& axes) {
        ov::reference::softmax(in.data<const T>(), out.data<T>(), shape, axes);
        return true;
    }
};
}  // namespace
}  // namespace softmax

namespace v1 {
Softmax::Softmax(const Output<Node>& arg, const size_t axis) : Op({arg}), m_axis(axis) {
    constructor_validate_and_infer_types();
}

bool Softmax::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_Softmax_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    return true;
}

void Softmax::validate_and_infer_types() {
    OV_OP_SCOPE(v1_Softmax_validate_and_infer_types);
    const auto& input_shape = get_input_partial_shape(0);
    if (input_shape.rank().is_static())
        NODE_VALIDATION_CHECK(this,
                              m_axis < static_cast<size_t>(input_shape.rank().get_length()),
                              "Reduction axis (",
                              m_axis,
                              ") is out of bounds (argument shape: ",
                              input_shape,
                              ").");

    set_output_type(0, get_input_element_type(0), input_shape);
}

std::shared_ptr<Node> Softmax::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Softmax_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Softmax>(new_args.at(0), m_axis);
}

bool Softmax::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_Softmax_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 1);

    const auto& input_shape = inputs[0].get_shape();
    outputs[0].set_shape(input_shape);
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v1_Softmax_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, f64),
                                      softmax::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      input_shape,
                                      AxisSet{m_axis});
}

bool Softmax::has_evaluate() const {
    OV_OP_SCOPE(v1_Softmax_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::f64:
        return true;
    default:
        return false;
    }
}
}  // namespace v1

namespace v8 {
Softmax::Softmax(const Output<Node>& arg, const int64_t axis) : Op({arg}), m_axis(axis) {
    constructor_validate_and_infer_types();
}

bool Softmax::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v8_Softmax_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    return true;
}

void Softmax::validate_and_infer_types() {
    OV_OP_SCOPE(v8_Softmax_validate_and_infer_types);
    const auto& input_shape = get_input_partial_shape(0);
    if (input_shape.rank().is_static()) {
        auto rank = static_cast<int64_t>(input_shape.size());
        NODE_VALIDATION_CHECK(this,
                              -rank <= m_axis && m_axis < rank,
                              "Reduction axis (",
                              m_axis,
                              ") is out of bounds (argument shape: ",
                              input_shape,
                              ").");
    }

    set_output_type(0, get_input_element_type(0), input_shape);
}

std::shared_ptr<Node> Softmax::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v8_Softmax_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Softmax>(new_args.at(0), m_axis);
}

bool Softmax::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v8_Softmax_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 1);

    const auto& input_shape = inputs[0].get_shape();
    const auto rank = static_cast<int64_t>(input_shape.size());
    OPENVINO_ASSERT(-rank <= m_axis && m_axis < rank,
                    "Reduction axis (",
                    m_axis,
                    ") is out of bounds (argument shape: ",
                    input_shape,
                    ").");
    const auto axis = static_cast<size_t>(ov::util::normalize(m_axis, rank));

    outputs[0].set_shape(input_shape);
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v8_Softmax_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, f64),
                                      softmax::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      outputs[0],
                                      input_shape,
                                      AxisSet{axis});
}

bool Softmax::has_evaluate() const {
    OV_OP_SCOPE(v8_Softmax_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::f64:
        return true;
    default:
        return false;
    }
}
}  // namespace v8
}  // namespace op
}  // namespace ov
