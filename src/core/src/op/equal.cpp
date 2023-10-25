// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/equal.hpp"

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/reference/equal.hpp"

using namespace std;
using namespace ngraph;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace equal {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& arg0,
              const HostTensorPtr& arg1,
              const HostTensorPtr& out,
              const op::AutoBroadcastSpec& broadcast_spec) {
    ov::reference::equal(arg0->get_data_ptr<ET>(),
                         arg1->get_data_ptr<ET>(),
                         out->get_data_ptr<element::Type_t::boolean>(),
                         arg0->get_shape(),
                         arg1->get_shape(),
                         broadcast_spec);
    return true;
}

bool evaluate_equal(const HostTensorPtr& arg0,
                    const HostTensorPtr& arg1,
                    const HostTensorPtr& out,
                    const op::AutoBroadcastSpec& broadcast_spec) {
    bool rc = true;
    out->set_broadcast(broadcast_spec, arg0, arg1, element::boolean);
    switch (arg0->get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_equal, boolean, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_equal, i4, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_equal, i8, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_equal, i16, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_equal, i32, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_equal, i64, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_equal, u4, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_equal, u8, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_equal, u16, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_equal, u32, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_equal, u64, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_equal, bf16, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_equal, f16, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_equal, f32, arg0, arg1, out, broadcast_spec);
        OPENVINO_TYPE_CASE(evaluate_equal, f64, arg0, arg1, out, broadcast_spec);
    default:
        rc = false;
        break;
    }
    return rc;
}

ov::Tensor equal_tensor(const ov::Tensor& lhs, const ov::Tensor& rhs) {
    auto equal = op::v1::Equal(std::make_shared<op::v0::Parameter>(lhs.get_element_type(), lhs.get_shape()),
                               std::make_shared<op::v0::Parameter>(rhs.get_element_type(), rhs.get_shape()),
                               op::AutoBroadcastType::NUMPY);
    auto outs = ov::TensorVector{{equal.get_output_element_type(0), equal.get_output_shape(0)}};
    equal.evaluate(outs, ov::TensorVector{lhs, rhs});
    return outs.front();
}

ov::Tensor less_equal_tensor(const ov::Tensor& lhs, const ov::Tensor& rhs) {
    auto equal = op::v1::LessEqual(std::make_shared<op::v0::Parameter>(lhs.get_element_type(), lhs.get_shape()),
                                   std::make_shared<op::v0::Parameter>(rhs.get_element_type(), rhs.get_shape()),
                                   op::AutoBroadcastType::NUMPY);
    auto outs = ov::TensorVector{{equal.get_output_element_type(0), equal.get_output_shape(0)}};
    equal.evaluate(outs, ov::TensorVector{lhs, rhs});
    return outs.front();
}

ov::Tensor and_tensor(const ov::Tensor& lhs, const ov::Tensor& rhs) {
    auto logical_and =
        ov::op::v1::LogicalAnd(std::make_shared<op::v0::Parameter>(lhs.get_element_type(), lhs.get_shape()),
                               std::make_shared<op::v0::Parameter>(rhs.get_element_type(), rhs.get_shape()),
                               op::AutoBroadcastType::NUMPY);
    auto outs = ov::TensorVector{{logical_and.get_output_element_type(0), logical_and.get_output_shape(0)}};
    logical_and.evaluate(outs, ov::TensorVector{lhs, rhs});
    return outs.front();
}

ov::Tensor or_tensor(const ov::Tensor& lhs, const ov::Tensor& rhs) {
    auto logical_or =
        ov::op::v1::LogicalOr(std::make_shared<op::v0::Parameter>(lhs.get_element_type(), lhs.get_shape()),
                              std::make_shared<op::v0::Parameter>(rhs.get_element_type(), rhs.get_shape()),
                              op::AutoBroadcastType::NUMPY);
    auto outs = ov::TensorVector{{logical_or.get_output_element_type(0), logical_or.get_output_shape(0)}};
    logical_or.evaluate(outs, ov::TensorVector{lhs, rhs});
    return outs.front();
}

void all_equal(const ov::TensorVector tensors, ov::Tensor& output_value) {
    OPENVINO_ASSERT(tensors.size() >= 2, "Unexpected number of tensors in all_equal helper");
    auto& tensor = tensors[0];
    output_value = equal_tensor(tensor, tensors[1]);
    for (size_t i = 2; i < tensors.size(); ++i) {
        output_value = and_tensor(output_value, equal_tensor(tensor, tensors[i]));
    }
}

ov::Tensor within_interval(const ov::Tensor& lower, const ov::Tensor& upper, const ov::Tensor& subject_to_check) {
    auto lower_check = less_equal_tensor(lower, subject_to_check);
    auto upper_check = less_equal_tensor(subject_to_check, upper);
    return and_tensor(lower_check, upper_check);
}

}  // namespace
}  // namespace equal

//------------------------------- v1 -------------------------------------------
op::v1::Equal::Equal(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseComparison(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::Equal::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Equal_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v1::Equal>(new_args.at(0), new_args.at(1), this->get_autob());
}

bool op::v1::Equal::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v1_Equal_evaluate);
    return equal::evaluate_equal(inputs[0], inputs[1], outputs[0], get_autob());
}

bool op::v1::Equal::evaluate_lower(ov::TensorVector& output_values) const {
    if (get_input_tensor(0).has_and_set_bound() && get_input_tensor(1).has_and_set_bound())
        return default_upper_bound_evaluator(this, output_values);
    // ll == lu == rl == ru     -> {true}
    // else                     -> {false}
    const auto &lhs = get_input_tensor(0), &rhs = get_input_tensor(1);
    auto lhs_lower = lhs.get_lower_value(), lhs_upper = lhs.get_upper_value();
    auto rhs_lower = rhs.get_lower_value(), rhs_upper = rhs.get_upper_value();
    equal::all_equal({lhs_lower, lhs_upper, rhs_lower, rhs_upper}, output_values[0]);
    return true;
}

bool op::v1::Equal::evaluate_upper(ov::TensorVector& output_values) const {
    const auto &lhs = get_input_tensor(0), &rhs = get_input_tensor(1);
    auto lhs_lower = lhs.get_lower_value(), lhs_upper = lhs.get_upper_value();
    auto rhs_lower = rhs.get_lower_value(), rhs_upper = rhs.get_upper_value();
    // check for intersection:
    // ll <= rl <= lu or ll <= ru <= lu
    auto rl_check = equal::within_interval(lhs_lower, lhs_upper, rhs_lower);
    auto ru_check = equal::within_interval(lhs_lower, lhs_upper, rhs_upper);
    output_values[0] = equal::or_tensor(rl_check, ru_check);
    return true;
}

bool op::v1::Equal::has_evaluate() const {
    OV_OP_SCOPE(v1_Equal_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::boolean:
    case ngraph::element::i8:
    case ngraph::element::u8:
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

bool op::v1::Equal::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_Equal_visit_attributes);
    BinaryElementwiseComparison::visit_attributes(visitor);
    return true;
}
