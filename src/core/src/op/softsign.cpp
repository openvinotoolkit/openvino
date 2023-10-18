// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/softsign.hpp"

#include <openvino/core/validation_util.hpp>

#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/reference/softsign.hpp"
#include "openvino/runtime/tensor.hpp"

namespace {
template <ov::element::Type_t ET>
inline bool evaluate(const ov::Tensor& arg, const ov::Tensor& out, const size_t count) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ov::reference::softsign<T>(arg.data<T>(), out.data<T>(), count);
    return true;
}

bool evaluate_softsign(const ov::Tensor& arg, const ov::Tensor& out) {
    bool rc = true;
    size_t count = arg.get_size();

    switch (arg.get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_softsign, bf16, arg, out, count);
        OPENVINO_TYPE_CASE(evaluate_softsign, f16, arg, out, count);
        OPENVINO_TYPE_CASE(evaluate_softsign, f32, arg, out, count);
        OPENVINO_TYPE_CASE(evaluate_softsign, f64, arg, out, count);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace

ov::op::v9::SoftSign::SoftSign(const Output<Node>& arg) : UnaryElementwiseArithmetic(arg) {
    constructor_validate_and_infer_types();
}

void ov::op::v9::SoftSign::validate_and_infer_types() {
    OV_OP_SCOPE(v9_SoftSign_validate_and_infer_types);
    const element::Type& input_et = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_et.is_dynamic() || input_et.is_real(),
                          "Input element type must be float, instead got: ",
                          input_et);

    UnaryElementwiseArithmetic::validate_and_infer_types();
}

bool ov::op::v9::SoftSign::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v9_SoftSign_visit_attributes);
    return true;
}

std::shared_ptr<ov::Node> ov::op::v9::SoftSign::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v9_SoftSign_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::op::v9::SoftSign>(new_args.at(0));
}

bool ov::op::v9::SoftSign::has_evaluate() const {
    OV_OP_SCOPE(v9_SoftSign_has_evaluate);
    switch (get_input_element_type(0)) {
    case ov::element::bf16:
    case ov::element::f16:
    case ov::element::f32:
    case ov::element::f64:
        return true;
    default:
        break;
    }
    return false;
}

bool ov::op::v9::SoftSign::evaluate(ov::TensorVector& outputs,
                                    const ov::TensorVector& inputs,
                                    const ov::EvaluationContext& evaluation_context) const {
    OV_OP_SCOPE(v9_SoftSign_evaluate);

    OPENVINO_ASSERT(outputs.size() == 1 && inputs.size() == 1,
                    "SoftSign evaluate needs exactly 1 input and 1 output, instead got:",
                    inputs.size(),
                    " input(s) and ",
                    outputs.size(),
                    " output(s).");

    const auto& in = inputs[0];
    auto& out = outputs[0];

    out.set_shape(in.get_shape());
    return evaluate_softsign(in, out);
}
