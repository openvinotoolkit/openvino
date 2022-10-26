// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/is_finite.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/runtime/reference/is_finite.hpp"

namespace ov {
ov::op::v10::IsFinite::IsFinite(const Output<Node>& data) : op::Op{{data}} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::op::v10::IsFinite::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v10_IsFinite_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<IsFinite>(new_args.at(0));
}

void ov::op::v10::IsFinite::validate_and_infer_types() {
    OV_OP_SCOPE(v10_IsFinite_validate_and_infer_types);
    element::Type input_element_type = get_input_element_type(0);
    element::Type output_element_type = ov::element::boolean;
    ov::PartialShape input_pshape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "The element type of the input tensor must be a floating point number or dynamic (got ",
                          input_element_type,
                          ").");
    set_output_type(0, output_element_type, input_pshape);
}

bool ov::op::v10::IsFinite::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v10_IsFinite_visit_attributes);
    return true;
}

namespace {
template <element::Type_t ET>
bool evaluate_exec(const HostTensorPtr& input, const HostTensorPtr& output) {
    ngraph::runtime::reference::is_finite(input->get_data_ptr<ET>(),
                                          output->get_data_ptr<element::Type_t::boolean>(),
                                          shape_size(input->get_shape()));
    return true;
}

#define IS_NAN_TYPE_CASE(a, ...)                                \
    case element::Type_t::a: {                                  \
        OV_OP_SCOPE(OV_PP_CAT3(evaluate_exec_is_finite, _, a)); \
        rc = evaluate_exec<element::Type_t::a>(__VA_ARGS__);    \
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

bool evaluate_is_finite(const HostTensorPtr& input, const HostTensorPtr& output) {
    bool rc = true;
    switch (input->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_is_finite, bf16, input, output);
        NGRAPH_TYPE_CASE(evaluate_is_finite, f16, input, output);
        NGRAPH_TYPE_CASE(evaluate_is_finite, f32, input, output);
        NGRAPH_TYPE_CASE(evaluate_is_finite, f64, input, output);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace

bool op::v10::IsFinite::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v10_IsFinite_evaluate);
    OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(inputs, 1), "Invalid IsFinite input TensorVector.");
    OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(outputs, 1), "Invalid IsFinite output TensorVector.");
    return evaluate_is_finite(inputs[0], outputs[0]);
}

bool op::v10::IsFinite::has_evaluate() const {
    OV_OP_SCOPE(v10_IsFinite_has_evaluate);
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
