// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/gelu.hpp"

#include <cmath>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/runtime/reference/gelu.hpp"

using namespace std;
using namespace ngraph;

// ------------------------------ V0 ------------------------------
NGRAPH_RTTI_DEFINITION(op::v0::Gelu, "Gelu", 0);

op::v0::Gelu::Gelu() : Op() {}

op::v0::Gelu::Gelu(const Output<Node>& data) : Op({data}) {
    constructor_validate_and_infer_types();
}

bool op::v0::Gelu::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v0_Gelu_visit_attributes);
    return true;
}

shared_ptr<Node> op::v0::Gelu::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v0_Gelu_clone_with_new_inputs);
    if (new_args.size() != 1) {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<op::v0::Gelu>(new_args.at(0));
}

void op::v0::Gelu::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v0_Gelu_validate_and_infer_types);
    element::Type input_element_type = get_input_element_type(0);
    PartialShape input_pshape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");

    set_output_type(0, input_element_type, input_pshape);
}

// ------------------------------ V7 ------------------------------

namespace ov {
template <>
NGRAPH_API EnumNames<op::GeluApproximationMode>& EnumNames<op::GeluApproximationMode>::get() {
    static auto enum_names = EnumNames<op::GeluApproximationMode>(
        "op::GeluApproximationMode",
        {{"TANH", op::GeluApproximationMode::TANH}, {"ERF", op::GeluApproximationMode::ERF}});
    return enum_names;
}

constexpr DiscreteTypeInfo AttributeAdapter<op::GeluApproximationMode>::type_info;
}  // namespace ov
std::ostream& op::operator<<(std::ostream& s, const op::GeluApproximationMode& type) {
    return s << as_string(type);
}

NGRAPH_RTTI_DEFINITION(op::v7::Gelu, "Gelu", 7);

op::v7::Gelu::Gelu(const Output<Node>& data, GeluApproximationMode mode)
    : UnaryElementwiseArithmetic(data),
      m_approximation_mode(mode) {
    constructor_validate_and_infer_types();
}

bool op::v7::Gelu::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v7_Gelu_visit_attributes);
    visitor.on_attribute("approximation_mode", m_approximation_mode);
    return true;
}

shared_ptr<Node> op::v7::Gelu::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v7_Gelu_clone_with_new_inputs);
    if (new_args.size() != 1) {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<op::v7::Gelu>(new_args.at(0), m_approximation_mode);
}

void op::v7::Gelu::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v7_Gelu_validate_and_infer_types);
    element::Type input_element_type = get_input_element_type(0);
    PartialShape input_pshape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");

    set_output_type(0, input_element_type, input_pshape);
}

op::GeluApproximationMode op::v7::Gelu::get_approximation_mode() const {
    return m_approximation_mode;
}

namespace gelu {
template <element::Type_t ET>
inline bool evaluate(const HostTensorPtr& arg0,
                     const HostTensorPtr& out,
                     op::GeluApproximationMode mode,
                     const size_t count) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::gelu<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), mode, count);
    return true;
}

bool evaluate_gelu(const HostTensorPtr& arg0, const HostTensorPtr& out, op::GeluApproximationMode mode) {
    bool rc = true;
    size_t count = shape_size(arg0->get_shape());
    out->set_unary(arg0);

    switch (arg0->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_gelu, f16, arg0, out, mode, count);
        NGRAPH_TYPE_CASE(evaluate_gelu, f32, arg0, out, mode, count);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace gelu

bool op::v7::Gelu::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    NGRAPH_OP_SCOPE(v7_Gelu_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, 1) && validate_host_tensor_vector(inputs, 1));
    return gelu::evaluate_gelu(inputs[0], outputs[0], m_approximation_mode);
}

bool op::v7::Gelu::has_evaluate() const {
    NGRAPH_OP_SCOPE(v7_Gelu_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::f16:
    case ngraph::element::f32:
        return true;
    default:
        break;
    }
    return false;
}
