// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/atanh.hpp"

#include <string>
#include <vector>

#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/atanh.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(ov::op::v3::Atanh);

op::v3::Atanh::Atanh(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v3::Atanh::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_Atanh_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Atanh>(new_args.at(0));
}

namespace atanhop {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& out) {
    runtime::reference::atanh(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), shape_size(arg0->get_shape()));
    return true;
}

bool evaluate_atanh(const HostTensorPtr& arg0, const HostTensorPtr& out) {
    bool rc = true;
    out->set_unary(arg0);
    switch (arg0->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_atanh, i32, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_atanh, i64, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_atanh, u32, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_atanh, u64, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_atanh, f16, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_atanh, f32, arg0, out);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace atanhop

bool op::v3::Atanh::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v3_Atanh_evaluate);
    return atanhop::evaluate_atanh(inputs[0], outputs[0]);
}

bool op::v3::Atanh::has_evaluate() const {
    OV_OP_SCOPE(v1_Atanh_has_evaluate);
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
