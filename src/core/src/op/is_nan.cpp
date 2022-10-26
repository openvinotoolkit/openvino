// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/is_nan.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/runtime/reference/is_nan.hpp"

namespace ov {
ov::op::v10::IsNaN::IsNaN(const Output<Node>& data) : op::Op{{data}} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::op::v10::IsNaN::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v10_IsNaN_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<IsNaN>(new_args.at(0));
}

void ov::op::v10::IsNaN::validate_and_infer_types() {
    OV_OP_SCOPE(v10_IsNaN_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).is_dynamic() || get_input_element_type(0).is_real(),
                          "The element type of the input tensor must be a floating point number.");
    set_output_type(0, ov::element::Type_t::boolean, get_input_partial_shape(0));
}

bool ov::op::v10::IsNaN::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v10_IsNaN_visit_attributes);
    return true;
}

namespace {
template <element::Type_t ET>
bool evaluate_exec(const HostTensorPtr& input, const HostTensorPtr& output) {
    ngraph::runtime::reference::is_nan(input->get_data_ptr<ET>(),
                                       output->get_data_ptr<element::Type_t::boolean>(),
                                       shape_size(input->get_shape()));
    return true;
}

#define IS_NAN_TYPE_CASE(a, ...)                             \
    case element::Type_t::a: {                               \
        OV_OP_SCOPE(OV_PP_CAT3(evaluate_exec_is_nan, _, a)); \
        rc = evaluate_exec<element::Type_t::a>(__VA_ARGS__); \
    } break

template <element::Type_t ET>
bool evaluate(const HostTensorPtr& input, const HostTensorPtr& output) {
    bool rc = true;
    switch (input->get_element_type()) {
        IS_NAN_TYPE_CASE(bf16, input, output);
        IS_NAN_TYPE_CASE(f16, input, output);
        IS_NAN_TYPE_CASE(f32, input, output);
        IS_NAN_TYPE_CASE(f64, input, output);
    default:
        rc = false;
        break;
    }
    return rc;
}

bool evaluate_is_nan(const HostTensorPtr& input, const HostTensorPtr& output) {
    bool rc = true;
    switch (input->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_is_nan, bf16, input, output);
        NGRAPH_TYPE_CASE(evaluate_is_nan, f16, input, output);
        NGRAPH_TYPE_CASE(evaluate_is_nan, f32, input, output);
        NGRAPH_TYPE_CASE(evaluate_is_nan, f64, input, output);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace

bool op::v10::IsNaN::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v10_IsNaN_evaluate);
    OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(inputs, 1), "Invalid IsNaN input TensorVector.");
    OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(outputs, 1), "Invalid IsNaN output TensorVector.");
    return evaluate_is_nan(inputs[0], outputs[0]);
}

bool op::v10::IsNaN::has_evaluate() const {
    OV_OP_SCOPE(v10_IsNaN_has_evaluate);
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
