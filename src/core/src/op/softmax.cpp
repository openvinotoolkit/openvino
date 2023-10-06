// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/softmax.hpp"

#include <algorithm>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "openvino/reference/softmax.hpp"

using namespace std;
using namespace ngraph;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace {
template <element::Type_t ET>
inline bool evaluate(const HostTensorPtr& arg, const HostTensorPtr& out, const ov::Shape& shape, const AxisSet& axes) {
    ov::reference::softmax(arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), shape, axes);
    return true;
}

bool evaluate_softmax(const HostTensorPtr& arg, const HostTensorPtr& out, const AxisSet& axes) {
    auto shape = out->get_shape();
    bool rc = true;

    switch (arg->get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_softmax, bf16, arg, out, shape, axes);
        OPENVINO_TYPE_CASE(evaluate_softmax, f16, arg, out, shape, axes);
        OPENVINO_TYPE_CASE(evaluate_softmax, f32, arg, out, shape, axes);
        OPENVINO_TYPE_CASE(evaluate_softmax, f64, arg, out, shape, axes);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace

// *** SOFTMAX OP SET V1 ***

op::v1::Softmax::Softmax(const Output<Node>& arg, const size_t axis) : Op({arg}), m_axis(axis) {
    constructor_validate_and_infer_types();
}

bool ngraph::op::v1::Softmax::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_Softmax_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    return true;
}

void op::v1::Softmax::validate_and_infer_types() {
    OV_OP_SCOPE(v1_Softmax_validate_and_infer_types);
    const ov::PartialShape& input_shape = get_input_partial_shape(0);
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

shared_ptr<Node> op::v1::Softmax::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Softmax_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v1::Softmax>(new_args.at(0), m_axis);
}

bool op::v1::Softmax::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v1_Softmax_evaluate);
    OPENVINO_SUPPRESS_DEPRECATED_START
    OPENVINO_ASSERT(validate_host_tensor_vector(outputs, 1) && validate_host_tensor_vector(inputs, 1));
    OPENVINO_SUPPRESS_DEPRECATED_END
    outputs[0]->set_unary(inputs[0]);
    return evaluate_softmax(inputs[0], outputs[0], AxisSet{m_axis});
}

bool op::v1::Softmax::has_evaluate() const {
    OV_OP_SCOPE(v1_Softmax_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::bf16:
    case ngraph::element::f16:
    case ngraph::element::f32:
    case ngraph::element::f64:
        return true;
    default:
        break;
    }
    return false;
}

// *** SOFTMAX OP SET V8 ***
op::v8::Softmax::Softmax(const Output<Node>& arg, const int64_t axis) : Op({arg}), m_axis(axis) {
    constructor_validate_and_infer_types();
}

bool op::v8::Softmax::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v8_Softmax_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    return true;
}

void op::v8::Softmax::validate_and_infer_types() {
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

shared_ptr<Node> op::v8::Softmax::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v8_Softmax_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v8::Softmax>(new_args.at(0), m_axis);
}

bool op::v8::Softmax::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v8_Softmax_evaluate);
    OPENVINO_SUPPRESS_DEPRECATED_START
    OPENVINO_ASSERT(validate_host_tensor_vector(outputs, 1) && validate_host_tensor_vector(inputs, 1));
    OPENVINO_SUPPRESS_DEPRECATED_END
    outputs[0]->set_unary(inputs[0]);
    auto rank = static_cast<int64_t>(inputs[0]->get_shape().size());
    OPENVINO_ASSERT(-rank <= m_axis && m_axis < rank,
                    "Reduction axis (",
                    m_axis,
                    ") is out of bounds (argument shape: ",
                    inputs[0]->get_shape(),
                    ").");
    OPENVINO_SUPPRESS_DEPRECATED_START
    size_t axis = static_cast<size_t>(ov::normalize_axis(this->description(), m_axis, rank));
    OPENVINO_SUPPRESS_DEPRECATED_END
    return evaluate_softmax(inputs[0], outputs[0], AxisSet{axis});
}

bool op::v8::Softmax::has_evaluate() const {
    OV_OP_SCOPE(v8_Softmax_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::bf16:
    case ngraph::element::f16:
    case ngraph::element::f32:
    case ngraph::element::f64:
        return true;
    default:
        break;
    }
    return false;
}
