// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/or.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v1::LogicalOr);

op::v1::LogicalOr::LogicalOr(const Output<Node>& arg0,
                             const Output<Node>& arg1,
                             const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseLogical(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::LogicalOr::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_LogicalOr_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::LogicalOr>(new_args.at(0), new_args.at(1), this->get_autob());
}

namespace logor {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& arg0,
              const HostTensorPtr& arg1,
              const HostTensorPtr& out,
              const op::AutoBroadcastSpec& broadcast_spec) {
    runtime::reference::logical_or(arg0->get_data_ptr<ET>(),
                                   arg1->get_data_ptr<ET>(),
                                   out->get_data_ptr<ET>(),
                                   arg0->get_shape(),
                                   arg1->get_shape(),
                                   broadcast_spec);
    return true;
}

bool evaluate_logor(const HostTensorPtr& arg0,
                    const HostTensorPtr& arg1,
                    const HostTensorPtr& out,
                    const op::AutoBroadcastSpec& broadcast_spec) {
    bool rc = true;
    out->set_broadcast(broadcast_spec, arg0, arg1);
    switch (arg0->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_logor, boolean, arg0, arg1, out, broadcast_spec);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace logor

bool op::v1::LogicalOr::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v1_LogicalOr_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(outputs, 1) && validate_host_tensor_vector(inputs, 2));
    return logor::evaluate_logor(inputs[0], inputs[1], outputs[0], get_autob());
}

bool op::v1::LogicalOr::has_evaluate() const {
    OV_OP_SCOPE(v1_LogicalOr_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::boolean:
        return true;
    default:
        break;
    }
    return false;
}
