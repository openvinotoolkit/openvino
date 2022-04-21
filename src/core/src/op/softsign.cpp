// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ngraph/runtime/reference/softsign.hpp"

#include <openvino/core/validation_util.hpp>

#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/op/softsign.hpp"
#include "openvino/runtime/tensor.hpp"

namespace {
template <ov::element::Type_t ET>
inline bool evaluate(const ov::Tensor& arg, const ov::Tensor& out, const size_t count) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::softsign<T>(arg.data<T>(), out.data<T>(), count);
    return true;
}

bool evaluate_softsign(const ov::Tensor& arg, const ov::Tensor& out) {
    bool rc = true;
    size_t count = arg.get_size();

    switch (arg.get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_softsign, bf16, arg, out, count);
        NGRAPH_TYPE_CASE(evaluate_softsign, f16, arg, out, count);
        NGRAPH_TYPE_CASE(evaluate_softsign, f32, arg, out, count);
        NGRAPH_TYPE_CASE(evaluate_softsign, f64, arg, out, count);
    default:
        rc = false;
        break;
    }
    return rc;
}

}  // namespace

BWDCMP_RTTI_DEFINITION(ov::op::v9::SoftSign);

ov::op::v9::SoftSign::SoftSign(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

bool ov::op::v9::SoftSign::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v9_SoftSign_visit_attributes);
    return true;
}

std::shared_ptr<ov::Node> ov::op::v9::SoftSign::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v9_SoftSign_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::op::v9::SoftSign>(new_args.at(0));
}

bool ov::op::v9::SoftSign::has_evaluate() const {
    NGRAPH_OP_SCOPE(v9_SoftSign_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::bf16:
    case ngraph::element::f16:
    case ngraph::element::f32:
    case ngraph::element::f64:
        return true;
    default:
        break;
    }
    return false;
}

bool ov::op::v9::SoftSign::evaluate(ov::TensorVector& outputs,
                                    const ov::TensorVector& inputs,
                                    const ov::EvaluationContext& evaluation_context) const {
    NGRAPH_OP_SCOPE(v9_SoftSign_evaluate);
    const auto& in = inputs[0];
    auto& out = outputs[0];
    out.set_shape(in.get_shape());
    return evaluate_softsign(in, out);
}
