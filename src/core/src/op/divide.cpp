// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/divide.hpp"

#include "bound_evaluate.hpp"
#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/select.hpp"
#include "openvino/reference/divide.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {
namespace divide {
namespace {
using ov::op::v0::Constant;
using ov::op::v0::Parameter;

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& in0,
                             const Tensor& in1,
                             Tensor& out,
                             const Shape& shape0,
                             const Shape& shape1,
                             const op::AutoBroadcastSpec& broadcast_spec,
                             const bool pythondiv) {
        reference::divide(in0.data<const T>(),
                          in1.data<const T>(),
                          out.data<T>(),
                          shape0,
                          shape1,
                          broadcast_spec,
                          pythondiv);
        return true;
    }
};

Tensor equality_mask(const Tensor& lhs, const Tensor& rhs) {
    auto mask_out = TensorVector{{element::boolean, lhs.get_shape()}};

    const auto lhs_node = std::make_shared<Parameter>(lhs.get_element_type(), lhs.get_shape());
    const auto rhs_node = std::make_shared<Parameter>(rhs.get_element_type(), rhs.get_shape());
    Equal(lhs_node, rhs_node).evaluate(mask_out, TensorVector{lhs, rhs});
    return mask_out.front();
}

Tensor or_tensor(const Tensor& lhs, const Tensor& rhs) {
    auto logical_or = LogicalOr(std::make_shared<Parameter>(lhs.get_element_type(), lhs.get_shape()),
                                std::make_shared<Parameter>(rhs.get_element_type(), rhs.get_shape()),
                                AutoBroadcastType::NUMPY);

    auto outs = TensorVector{{lhs.get_element_type(), logical_or.get_output_shape(0)}};
    logical_or.evaluate(outs, TensorVector{lhs, rhs});
    return outs.front();
}

bool evaluate_bound(const Node* node, TensorVector& output_values, bool is_upper) {
    // for positive arg2 divide will have limits [low/up , up/low]
    // for negative arg2 limits for divide will be [up/low, low/up]
    // for arg2 range with both positive and negative values, divide can give any result [-inf, inf]
    OPENVINO_ASSERT(node, output_values.size() == 1);
    const auto& input1 = node->input_value(0);
    const auto& input2 = node->input_value(1);

    // broadcast shapes to allocate tensors of correct size for operations with both inputs
    PartialShape input_shape = input1.get_partial_shape();
    OPENVINO_ASSERT(PartialShape::broadcast_merge_into(input_shape, input2.get_partial_shape(), node->get_autob()),
                    "Argument shapes in divide operation are inconsistent.");

    const auto input1_low = ov::util::evaluate_lower_bound(input1);
    if (!input1_low)
        return false;
    const auto input1_up = ov::util::evaluate_upper_bound(input1);
    if (!input1_up)
        return false;
    const auto input2_low = ov::util::evaluate_lower_bound(input2);
    if (!input2_low)
        return false;
    const auto input2_up = ov::util::evaluate_upper_bound(input2);
    if (!input2_up)
        return false;

    const auto zeros_const = Constant::create(input2.get_element_type(), {}, {0});
    const auto zero_t = Tensor(input2.get_element_type(), Shape{});
    memcpy(zero_t.data(), zeros_const->get_data_ptr(), zero_t.get_byte_size());

    const auto max_value = ov::util::make_tensor_of_max_value(input2.get_element_type());
    const auto dynamic_mask = or_tensor(equality_mask(input1_up, max_value), equality_mask(input2_up, max_value));

    // mask to find out positive values for arg2
    auto less_up_outputs = TensorVector{{element::boolean, input2.get_shape()}};
    auto& input2_positive_up_mask = less_up_outputs.front();

    bool status = Less().evaluate(less_up_outputs, TensorVector{zero_t, input2_up});
    if (!status)
        return status;

    // mask to find out negative values for arg2
    auto less_low_outputs = TensorVector{{element::boolean, input2.get_shape()}};
    auto& input2_negative_low_mask = less_low_outputs.front();
    status = Less().evaluate(less_low_outputs, {input2_low, zero_t});
    if (!status)
        return status;

    // mask to find out ranges around 0 for arg2
    auto logical_and_up_outputs = TensorVector{{element::boolean, input2.get_shape()}};
    auto& input2_low_negative_up_positive_mask = logical_and_up_outputs.front();
    status = LogicalAnd().evaluate(logical_and_up_outputs, {input2_negative_low_mask, input2_positive_up_mask});
    if (!status)
        return status;

    auto value1_outs = TensorVector{{input1.get_element_type(), input_shape.get_shape()}};
    auto& value1 = value1_outs.front();

    auto value2_outs = TensorVector{{input2.get_element_type(), input2.get_shape()}};
    auto& value2 = value2_outs.front();

    if (!is_upper) {
        status = Select().evaluate(value1_outs, {input2_positive_up_mask, input1_low, input1_up});
        if (!status)
            return status;

        status = Select().evaluate(value2_outs, {input2_positive_up_mask, input2_up, input2_low});
        if (!status)
            return status;

        status = node->evaluate(output_values, TensorVector{value1, value2});
        if (!status)
            return status;

        // replace values where zeros inside range of second arg to maximum values
        const auto output_min_value = ov::util::make_tensor_of_min_value(output_values[0].get_element_type());
        if (!output_min_value)
            return false;

        status = Select().evaluate(output_values,
                                   {input2_low_negative_up_positive_mask, output_min_value, output_values[0]});
        if (!status)
            return status;

        status = Select().evaluate(output_values, {dynamic_mask, zero_t, output_values[0]});
        if (!status)
            return status;
    } else {
        status = Select().evaluate(value1_outs, {input2_positive_up_mask, input1_up, input1_low});
        if (!status)
            return status;

        status = Select().evaluate(value2_outs, {input2_positive_up_mask, input2_low, input2_up});
        if (!status)
            return status;

        // create mask where zeros in the second argument are placed
        auto eq_zero_mask = TensorVector{{element::boolean, input2.get_shape()}};
        auto& input2_zeros_mask = eq_zero_mask.front();
        bool status = Equal().evaluate(eq_zero_mask, {value2, zero_t});
        if (!status)
            return status;

        // replace zeros by 1 values to get result of divide for other values of arguments
        const auto ones = Constant::create(input2.get_element_type(), input2.get_shape(), {1});
        const auto ones_t = Tensor(ones->get_element_type(), ones->get_shape());
        memcpy(ones_t.data(), ones->get_data_ptr(), ones_t.get_byte_size());

        status = Select().evaluate(value2_outs, {input2_zeros_mask, ones_t, value2});
        if (!status)
            return status;

        status = node->evaluate(output_values, {value1, value2});
        if (!status)
            return status;

        // replace values where zeros were found in the second argument to maximum values
        const auto out_max_value = ov::util::make_tensor_of_max_value(output_values[0].get_element_type());
        if (!out_max_value)
            return false;

        status = Select().evaluate(output_values, {input2_zeros_mask, out_max_value, output_values[0]});
        if (!status)
            return status;

        // replace values where zeros inside [low, ip] values range of second arg to maximum values
        status =
            Select().evaluate(output_values, {input2_low_negative_up_positive_mask, out_max_value, output_values[0]});
        if (!status)
            return status;

        // in case input elements were dynamic we replace them with zero
        status = Select().evaluate(output_values, {dynamic_mask, out_max_value, output_values[0]});
        if (!status)
            return status;
    }
    return status;
}
}  // namespace
}  // namespace divide

Divide::Divide(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

Divide::Divide(const Output<Node>& arg0,
               const Output<Node>& arg1,
               bool pythondiv,
               const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast),
      m_pythondiv(pythondiv) {
    constructor_validate_and_infer_types();
}

bool Divide::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_Divide_visit_attributes);
    BinaryElementwiseArithmetic::visit_attributes(visitor);
    visitor.on_attribute("m_pythondiv", m_pythondiv);
    return true;
}

std::shared_ptr<Node> Divide::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Divide_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Divide>(new_args.at(0), new_args.at(1), this->is_pythondiv(), this->get_autob());
}

bool Divide::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_Divide_evaluate);

    OPENVINO_ASSERT(outputs.size() == 1);

    outputs[0].set_shape(infer_broadcast_shape(this, inputs));
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v1_Divide_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(f32, i32, i64, u32, u64),
                                      divide::Evaluate,
                                      inputs[0].get_element_type(),
                                      inputs[0],
                                      inputs[1],
                                      outputs[0],
                                      inputs[0].get_shape(),
                                      inputs[1].get_shape(),
                                      get_autob(),
                                      is_pythondiv());
    return true;
}

bool Divide::has_evaluate() const {
    OV_OP_SCOPE(v1_Divide_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::i32:
    case element::i64:
    case element::u32:
    case element::u64:
    case element::f16:
    case element::bf16:
    case element::f32:
        return true;
    default:
        return false;
    }
}

bool Divide::evaluate_lower(TensorVector& outputs) const {
    return divide::evaluate_bound(this, outputs, false);
}

bool Divide::evaluate_upper(TensorVector& outputs) const {
    return divide::evaluate_bound(this, outputs, true);
}
}  // namespace v1
}  // namespace op
}  // namespace ov
