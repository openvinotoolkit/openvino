// Copyright (C) 2018-2025 Intel Corporation
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

void all_equal(const TensorVector& tensors, TensorVector& outputs) {
    auto& output = outputs[0];
    auto eq_result = TensorVector{{output.get_element_type(), output.get_shape()}};

    auto t_iter = tensors.begin() + 2;
    auto eq_inputs = TensorVector(tensors.begin(), t_iter);

    const auto eq = v1::Equal();
    eq.evaluate(outputs, eq_inputs);
    for (; t_iter != tensors.end(); ++t_iter) {
        eq_inputs[1] = *t_iter;
        eq.evaluate(eq_result, eq_inputs);
        output = and_tensor(output, eq_result[0]);
    }
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
    return IF_TYPE_OF_CONVERT_TENSORS(v1_Equal_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(boolean, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64),
                                      equal::Evaluate,
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
    equal::all_equal({lhs_lower, lhs_upper, rhs_lower, rhs_upper}, output_values);
    return true;
}

bool Equal::evaluate_upper(TensorVector& output_values) const {
    const auto &lhs = get_input_tensor(0), &rhs = get_input_tensor(1);
    const auto &lhs_lower = lhs.get_lower_value(), &lhs_upper = lhs.get_upper_value();
    const auto &rhs_lower = rhs.get_lower_value(), &rhs_upper = rhs.get_upper_value();

    // if (lhs_lower <= rhs_upper && rhs_lower <= lhs_upper) bounds have got intersection
    const auto lb_check = equal::less_equal_tensor(lhs_lower, rhs_upper);
    const auto ub_check = equal::less_equal_tensor(rhs_lower, lhs_upper);
    output_values[0] = equal::and_tensor(lb_check, ub_check);
    return true;
}

bool Equal::has_evaluate() const {
    OV_OP_SCOPE(v1_Equal_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::boolean:
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::f64:
    case element::i8:
    case element::i16:
    case element::i32:
    case element::i64:
    case element::u8:
    case element::u16:
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
