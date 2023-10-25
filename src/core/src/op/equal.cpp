// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/equal.hpp"

#include "bound_evaluate.hpp"
#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/reference/equal.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace equal {
namespace {

Tensor equal_tensor(const Tensor& lhs, const Tensor& rhs) {
    const auto eq = v1::Equal();
    auto outs = TensorVector{{element::boolean, Shape{}}};
    eq.evaluate(outs, {lhs, rhs});
    return outs.front();
}

Tensor less_equal_tensor(const Tensor& lhs, const Tensor& rhs) {
    const auto less_eq = v1::LessEqual();
    auto outs = TensorVector{{element::boolean, Shape{}}};
    less_eq.evaluate(outs, {lhs, rhs});
    return outs.front();
}

Tensor and_tensor(const Tensor& lhs, const Tensor& rhs) {
    const auto logical_and = v1::LogicalAnd();
    auto outs = TensorVector{{element::boolean, Shape{}}};
    logical_and.evaluate(outs, {lhs, rhs});
    return outs.front();
}

Tensor or_tensor(const Tensor& lhs, const Tensor& rhs) {
    const auto logical_or = v1::LogicalOr();
    auto outs = TensorVector{{element::boolean, Shape{}}};
    logical_or.evaluate(outs, {lhs, rhs});
    return outs.front();
}

void all_equal(const TensorVector tensors, Tensor& output_value) {
    auto& first_tensor = tensors[0];
    for (size_t i = 1; i < tensors.size(); ++i) {
        output_value = and_tensor(output_value, equal_tensor(first_tensor, tensors[i]));
    }
}

Tensor within_interval(const Tensor& lower, const Tensor& upper, const Tensor& subject_to_check) {
    const auto lower_check = less_equal_tensor(lower, subject_to_check);
    const auto upper_check = less_equal_tensor(subject_to_check, upper);
    return and_tensor(lower_check, upper_check);
}
}  // namespace

struct Evaluate : public element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& arg0,
                             const Tensor& arg1,
                             Tensor& out,
                             const Shape& shape0,
                             const Shape& shape1,
                             const op::AutoBroadcastSpec& broadcast_spec) {
        reference::equal(arg0.data<const T>(),
                         arg1.data<const T>(),
                         out.data<fundamental_type_for<element::boolean>>(),
                         shape0,
                         shape1,
                         broadcast_spec);
        return true;
    }
};
}  // namespace equal

//------------------------------- v1 -------------------------------------------
namespace v1 {
Equal::Equal(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseComparison(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> Equal::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Equal_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Equal>(new_args.at(0), new_args.at(1), get_autob());
}

bool Equal::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_Equal_evaluate);

    outputs[0].set_shape(ov::op::infer_broadcast_shape(this, inputs));
    using namespace ov::element;
    return IfTypeOf<boolean, bf16, f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64>::apply<equal::Evaluate>(
        inputs[0].get_element_type(),
        inputs[0],
        inputs[1],
        outputs[0],
        inputs[0].get_shape(),
        inputs[1].get_shape(),
        get_autob());
}

bool Equal::evaluate_lower(TensorVector& output_values) const {
    if (get_input_tensor(0).has_and_set_bound() && get_input_tensor(1).has_and_set_bound())
        return default_upper_bound_evaluator(this, output_values);
    // ll == lu == rl == ru     -> {true}
    // else                     -> {false}
    const auto &lhs = get_input_tensor(0), &rhs = get_input_tensor(1);
    const auto &lhs_lower = lhs.get_lower_value(), &lhs_upper = lhs.get_upper_value();
    const auto &rhs_lower = rhs.get_lower_value(), &rhs_upper = rhs.get_upper_value();
    equal::all_equal({lhs_lower, lhs_upper, rhs_lower, rhs_upper}, output_values[0]);
    return true;
}

bool Equal::evaluate_upper(TensorVector& output_values) const {
    const auto &lhs = get_input_tensor(0), &rhs = get_input_tensor(1);
    const auto &lhs_lower = lhs.get_lower_value(), &lhs_upper = lhs.get_upper_value();
    const auto &rhs_lower = rhs.get_lower_value(), &rhs_upper = rhs.get_upper_value();
    // check for intersection:
    // ll <= rl <= lu or ll <= ru <= lu
    const auto rl_check = equal::within_interval(lhs_lower, lhs_upper, rhs_lower);
    const auto ru_check = equal::within_interval(lhs_lower, lhs_upper, rhs_upper);
    output_values[0] = equal::or_tensor(rl_check, ru_check);
    return true;
}

bool Equal::has_evaluate() const {
    OV_OP_SCOPE(v1_Equal_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::boolean:
    case element::f16:
    case element::f32:
    case element::i8:
    case element::i32:
    case element::i64:
    case element::u8:
    case element::u32:
    case element::u64:
        return true;
    default:
        return false;
    }
}
}  // namespace v1
}  // namespace op
}  // namespace ov
