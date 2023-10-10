// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/multiply.hpp"

#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/reference/multiply.hpp"

using namespace std;
using namespace ngraph;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace multiplyop {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& arg0,
              const HostTensorPtr& arg1,
              const HostTensorPtr& out,
              const op::AutoBroadcastSpec& broadcast_spec) {
    ov::reference::multiply(arg0->get_data_ptr<ET>(),
                            arg1->get_data_ptr<ET>(),
                            out->get_data_ptr<ET>(),
                            arg0->get_shape(),
                            arg1->get_shape(),
                            broadcast_spec);
    return true;
}

bool evaluate_multiply(const HostTensorPtr& arg0,
                       const HostTensorPtr& arg1,
                       const HostTensorPtr& out,
                       const op::AutoBroadcastSpec& broadcast_spec) {
    bool rc = true;
    out->set_broadcast(broadcast_spec, arg0, arg1);
    switch (arg0->get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_multiply, i32, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_multiply, i64, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_multiply, u32, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_multiply, u64, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_multiply, f16, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_multiply, f32, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_multiply, bf16, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_multiply, u8, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_multiply, i16, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_multiply, u16, arg0, arg1, out, broadcast_spec);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace multiplyop

// ------------------------------------ v1 -------------------------------------
op::v1::Multiply::Multiply(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::Multiply::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Multiply_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v1::Multiply>(new_args.at(0), new_args.at(1), this->get_autob());
}

bool op::v1::Multiply::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v1_Multiply_evaluate);
    return multiplyop::evaluate_multiply(inputs[0], inputs[1], outputs[0], get_autob());
}

bool op::v1::Multiply::has_evaluate() const {
    OV_OP_SCOPE(v1_Multiply_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32:
    case ngraph::element::bf16:
        return true;
    default:
        break;
    }
    return false;
}
