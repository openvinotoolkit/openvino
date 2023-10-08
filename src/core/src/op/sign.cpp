// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/sign.hpp"

#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "openvino/reference/sign.hpp"

using namespace std;
using namespace ngraph;

op::Sign::Sign(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::Sign::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Sign_visit_attributes);
    return true;
}

shared_ptr<Node> op::Sign::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Sign_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Sign>(new_args.at(0));
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace signop {
namespace {
template <element::Type_t ET>
inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count) {
    using T = typename element_type_traits<ET>::value_type;
    ov::reference::sign<T>(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), count);
    return true;
}

bool evaluate_sign(const HostTensorPtr& arg0, const HostTensorPtr& out, const size_t count) {
    bool rc = true;
    out->set_unary(arg0);

    switch (arg0->get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_sign, i32, arg0, out, count);
        OPENVINO_TYPE_CASE(evaluate_sign, i64, arg0, out, count);
        OPENVINO_TYPE_CASE(evaluate_sign, u32, arg0, out, count);
        OPENVINO_TYPE_CASE(evaluate_sign, u64, arg0, out, count);
        OPENVINO_TYPE_CASE(evaluate_sign, f16, arg0, out, count);
        OPENVINO_TYPE_CASE(evaluate_sign, f32, arg0, out, count);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace signop

bool op::Sign::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Sign_evaluate);
    OPENVINO_SUPPRESS_DEPRECATED_START
    OPENVINO_ASSERT(validate_host_tensor_vector(outputs, 1) && validate_host_tensor_vector(inputs, 1));
    OPENVINO_SUPPRESS_DEPRECATED_END
    return signop::evaluate_sign(inputs[0], outputs[0], shape_size(inputs[0]->get_shape()));
}

bool op::Sign::has_evaluate() const {
    OV_OP_SCOPE(v0_Sign_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32:
        return true;
    default:
        break;
    }
    return false;
}
