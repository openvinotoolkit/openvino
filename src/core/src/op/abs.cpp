// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/abs.hpp"

#include "itt.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/abs.hpp"

BWDCMP_RTTI_DEFINITION(ov::op::v0::Abs);

ov::op::v0::Abs::Abs(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::op::v0::Abs::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Abs_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Abs>(new_args.at(0));
}

namespace absop {
namespace {
template <ov::element::Type_t ET>
inline bool evaluate(const ngraph::HostTensorPtr& arg0, const ngraph::HostTensorPtr& out, const size_t count) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::abs<T>((arg0->get_data_ptr<ET>()), (out->get_data_ptr<ET>()), count);
    return true;
}

bool evaluate_abs(const ngraph::HostTensorPtr& arg0, const ngraph::HostTensorPtr& out, const size_t count) {
    bool rc = true;
    out->set_unary(arg0);

    switch (arg0->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_abs, i32, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_abs, i64, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_abs, u32, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_abs, u64, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_abs, f16, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_abs, f32, arg0, out, count);
        NGRAPH_TYPE_CASE(evaluate_abs, bf16, arg0, out, count);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace absop

bool ov::op::v0::Abs::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v0_Abs_evaluate);
    return absop::evaluate_abs(inputs[0], outputs[0], shape_size(get_output_shape(0)));
}

bool ov::op::v0::Abs::has_evaluate() const {
    OV_OP_SCOPE(v0_Abs_has_evaluate);
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
