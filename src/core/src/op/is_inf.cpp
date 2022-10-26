// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/is_inf.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/runtime/reference/is_inf.hpp"

namespace ov {
op::v10::IsInf::IsInf(const Output<Node>& data) : op::Op{{data}} {
    constructor_validate_and_infer_types();
}

op::v10::IsInf::IsInf(const Output<Node>& data, const Attributes& attributes)
    : op::Op{{data}},
      m_attributes{attributes} {
    constructor_validate_and_infer_types();
}

bool op::v10::IsInf::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v10_IsInf_visit_attributes);
    visitor.on_attribute("detect_negative", m_attributes.detect_negative);
    visitor.on_attribute("detect_positive", m_attributes.detect_positive);
    return true;
}

void op::v10::IsInf::validate_and_infer_types() {
    OV_OP_SCOPE(v10_IsInf_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).is_dynamic() || get_input_element_type(0).is_real(),
                          "The element type of the input tensor must be a floating point number.");
    set_output_type(0, element::boolean, get_input_partial_shape(0));
}

std::shared_ptr<Node> op::v10::IsInf::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v10_IsInf_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v10::IsInf>(new_args.at(0), this->get_attributes());
}

namespace {
template <element::Type_t ET>
bool evaluate_exec(const HostTensorPtr& input,
                   const HostTensorPtr& output,
                   const op::v10::IsInf::Attributes& attributes) {
    ngraph::runtime::reference::is_inf(input->get_data_ptr<ET>(),
                                       output->get_data_ptr<element::Type_t::boolean>(),
                                       shape_size(input->get_shape()),
                                       attributes);
    return true;
}

#define IS_INF_TYPE_CASE(a, ...)                             \
    case element::Type_t::a: {                               \
        OV_OP_SCOPE(OV_PP_CAT3(evaluate_exec_is_inf, _, a)); \
        rc = evaluate_exec<element::Type_t::a>(__VA_ARGS__); \
    } break

template <element::Type_t ET>
bool evaluate(const HostTensorPtr& input, const HostTensorPtr& output, const op::v10::IsInf::Attributes& attributes) {
    bool rc = true;
    switch (input->get_element_type()) {
        IS_INF_TYPE_CASE(bf16, input, output, attributes);
        IS_INF_TYPE_CASE(f16, input, output, attributes);
        IS_INF_TYPE_CASE(f32, input, output, attributes);
        IS_INF_TYPE_CASE(f64, input, output, attributes);
    default:
        rc = false;
        break;
    }
    return rc;
}

bool evaluate_is_inf(const HostTensorPtr& input,
                     const HostTensorPtr& output,
                     const op::v10::IsInf::Attributes& attributes) {
    bool rc = true;
    switch (input->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_is_inf, bf16, input, output, attributes);
        NGRAPH_TYPE_CASE(evaluate_is_inf, f16, input, output, attributes);
        NGRAPH_TYPE_CASE(evaluate_is_inf, f32, input, output, attributes);
        NGRAPH_TYPE_CASE(evaluate_is_inf, f64, input, output, attributes);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace

bool op::v10::IsInf::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v10_IsInf_evaluate);
    OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(inputs, 1), "Invalid IsInf input TensorVector.");
    OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(outputs, 1), "Invalid IsInf output TensorVector.");
    return evaluate_is_inf(inputs[0], outputs[0], m_attributes);
}

bool op::v10::IsInf::has_evaluate() const {
    OV_OP_SCOPE(v10_IsInf_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::f64:
        return true;
    default:
        break;
    }
    return false;
}
}  // namespace ov
