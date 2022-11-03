// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/is_nan.hpp"

#include "itt.hpp"
#include "ngraph/runtime/reference/is_nan.hpp"
#include "utils.hpp"

namespace ov {
BWDCMP_RTTI_DEFINITION(op::v10::IsNaN);

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
bool evaluate_exec(const TensorVector& inputs, TensorVector& outputs) {
    using T = typename element_type_traits<ET>::value_type;
    using U = typename element_type_traits<element::Type_t::boolean>::value_type;
    ngraph::runtime::reference::is_nan(inputs[0].data<T>(), outputs[0].data<U>(), shape_size(inputs[0].get_shape()));
    return true;
}

#define IS_NAN_TYPE_CASE(a, ...)                             \
    case element::Type_t::a: {                               \
        OV_OP_SCOPE(OV_PP_CAT3(evaluate_exec_is_nan, _, a)); \
        rc = evaluate_exec<element::Type_t::a>(__VA_ARGS__); \
    } break

template <element::Type_t ET>
bool evaluate(const TensorVector& inputs, TensorVector& outputs) {
    bool rc = true;
    switch (inputs[0].get_element_type()) {
        IS_NAN_TYPE_CASE(bf16, inputs, outputs);
        IS_NAN_TYPE_CASE(f16, inputs, outputs);
        IS_NAN_TYPE_CASE(f32, inputs, outputs);
        IS_NAN_TYPE_CASE(f64, inputs, outputs);
    default:
        rc = false;
        break;
    }
    return rc;
}

bool evaluate_is_nan(const TensorVector& inputs, TensorVector& outputs) {
    bool rc = true;
    switch (inputs[0].get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_is_nan, bf16, inputs, outputs);
        NGRAPH_TYPE_CASE(evaluate_is_nan, f16, inputs, outputs);
        NGRAPH_TYPE_CASE(evaluate_is_nan, f32, inputs, outputs);
        NGRAPH_TYPE_CASE(evaluate_is_nan, f64, inputs, outputs);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace

bool op::v10::IsNaN::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v10_IsNaN_evaluate);
    OPENVINO_ASSERT(validate_tensor_vector(inputs, 1), "Invalid IsNaN input TensorVector.");
    OPENVINO_ASSERT(validate_tensor_vector(outputs, 1), "Invalid IsNaN output TensorVector.");
    return evaluate_is_nan(inputs, outputs);
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
