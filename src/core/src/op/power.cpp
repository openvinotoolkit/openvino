// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/power.hpp"

#include "itt.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/power.hpp"

using namespace std;
using namespace ngraph;

namespace power {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& arg0,
              const HostTensorPtr& arg1,
              const HostTensorPtr& out,
              const op::AutoBroadcastSpec& broadcast_spec) {
    runtime::reference::power(arg0->get_data_ptr<ET>(),
                              arg1->get_data_ptr<ET>(),
                              out->get_data_ptr<ET>(),
                              arg0->get_shape(),
                              arg1->get_shape(),
                              broadcast_spec);
    return true;
}

bool evaluate_power(const HostTensorPtr& arg0,
                    const HostTensorPtr& arg1,
                    const HostTensorPtr& out,
                    const op::AutoBroadcastSpec& broadcast_spec) {
    bool rc = true;
    out->set_broadcast(broadcast_spec, arg0, arg1);
    switch (arg0->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_power, i32, arg0, arg1, out, broadcast_spec);
        NGRAPH_TYPE_CASE(evaluate_power, i64, arg0, arg1, out, broadcast_spec);
        NGRAPH_TYPE_CASE(evaluate_power, u32, arg0, arg1, out, broadcast_spec);
        NGRAPH_TYPE_CASE(evaluate_power, u64, arg0, arg1, out, broadcast_spec);
        NGRAPH_TYPE_CASE(evaluate_power, f16, arg0, arg1, out, broadcast_spec);
        NGRAPH_TYPE_CASE(evaluate_power, f32, arg0, arg1, out, broadcast_spec);
        NGRAPH_TYPE_CASE(evaluate_power, bf16, arg0, arg1, out, broadcast_spec);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace power

// ------------------------------ v1 -------------------------------------------

BWDCMP_RTTI_DEFINITION(op::v1::Power);

op::v1::Power::Power(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::Power::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Power_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v1::Power>(new_args.at(0), new_args.at(1), this->get_autob());
}

bool op::v1::Power::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v1_Power_evaluate);
    return power::evaluate_power(inputs[0], inputs[1], outputs[0], get_autob());
}

bool op::v1::Power::has_evaluate() const {
    OV_OP_SCOPE(v1_Power_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::bf16:
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
