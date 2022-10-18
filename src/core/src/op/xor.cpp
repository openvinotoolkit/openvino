// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/xor.hpp"

#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/xor.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v1::LogicalXor);

op::v1::LogicalXor::LogicalXor(const Output<Node>& arg0,
                               const Output<Node>& arg1,
                               const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseLogical(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::LogicalXor::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_LogicalXor_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::LogicalXor>(new_args.at(0), new_args.at(1), this->get_autob());
}

namespace logxor {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& arg0,
              const HostTensorPtr& arg1,
              const HostTensorPtr& out,
              const op::AutoBroadcastSpec& broadcast_spec) {
    runtime::reference::logical_xor(arg0->get_data_ptr<ET>(),
                                    arg1->get_data_ptr<ET>(),
                                    out->get_data_ptr<ET>(),
                                    arg0->get_shape(),
                                    arg1->get_shape(),
                                    broadcast_spec);
    return true;
}

bool evaluate_logxor(const HostTensorPtr& arg0,
                     const HostTensorPtr& arg1,
                     const HostTensorPtr& out,
                     const op::AutoBroadcastSpec& broadcast_spec) {
    bool rc = true;
    out->set_broadcast(broadcast_spec, arg0, arg1);
    switch (arg0->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_logxor, boolean, arg0, arg1, out, broadcast_spec);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace logxor

bool op::v1::LogicalXor::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v1_LogicalXor_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, 1) && validate_host_tensor_vector(inputs, 2));
    return logxor::evaluate_logxor(inputs[0], inputs[1], outputs[0], get_autob());
}

bool op::v1::LogicalXor::has_evaluate() const {
    OV_OP_SCOPE(v1_LogicalXor_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::boolean:
        return true;
    default:
        break;
    }
    return false;
}

BWDCMP_RTTI_DEFINITION(op::v0::Xor);

op::v0::Xor::Xor(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseLogical(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v0::Xor::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Xor_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v0::Xor>(new_args.at(0), new_args.at(1), this->get_autob());
}

bool op::v0::Xor::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Xor_evaluate);
    return logxor::evaluate_logxor(inputs[0], inputs[1], outputs[0], get_autob());
}

bool op::v0::Xor::has_evaluate() const {
    OV_OP_SCOPE(v0_Xor_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::boolean:
        return true;
    default:
        break;
    }
    return false;
}
