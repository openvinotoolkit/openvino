// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/acosh.hpp"

#include <string>
#include <vector>

#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/acosh.hpp"
#include "ngraph/type/element_type.hpp"

BWDCMP_RTTI_DEFINITION(ov::op::v3::Acosh);

ov::op::v3::Acosh::Acosh(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::op::v3::Acosh::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_Acosh_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Acosh>(new_args.at(0));
}

namespace acoshop {
namespace {
template <ov::element::Type_t ET>
bool evaluate(const ngraph::HostTensorPtr& arg0, const ngraph::HostTensorPtr& out) {
    ngraph::runtime::reference::acosh(arg0->get_data_ptr<ET>(), out->get_data_ptr<ET>(), shape_size(arg0->get_shape()));
    return true;
}

bool evaluate_acosh(const ngraph::HostTensorPtr& arg0, const ngraph::HostTensorPtr& out) {
    bool rc = true;
    out->set_unary(arg0);
    switch (arg0->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_acosh, i32, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_acosh, i64, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_acosh, u32, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_acosh, u64, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_acosh, f16, arg0, out);
        NGRAPH_TYPE_CASE(evaluate_acosh, f32, arg0, out);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace acoshop

bool ov::op::v3::Acosh::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v3_Acosh_evaluate);
    return acoshop::evaluate_acosh(inputs[0], outputs[0]);
}

bool ov::op::v3::Acosh::has_evaluate() const {
    OV_OP_SCOPE(v3_Acosh_has_evaluate);
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
