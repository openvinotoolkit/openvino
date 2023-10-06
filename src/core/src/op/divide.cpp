// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/divide.hpp"

#include <ngraph/validation_util.hpp>

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/reference/divide.hpp"

using namespace std;
using namespace ngraph;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace divide {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& arg0,
              const HostTensorPtr& arg1,
              const HostTensorPtr& out,
              const op::AutoBroadcastSpec& broadcast_spec,
              bool pythondiv) {
    ov::reference::divide(arg0->get_data_ptr<ET>(),
                          arg1->get_data_ptr<ET>(),
                          out->get_data_ptr<ET>(),
                          arg0->get_shape(),
                          arg1->get_shape(),
                          broadcast_spec,
                          pythondiv);
    return true;
}

bool evaluate_divide(const HostTensorPtr& arg0,
                     const HostTensorPtr& arg1,
                     const HostTensorPtr& out,
                     const op::AutoBroadcastSpec& broadcast_spec,
                     bool pythondiv) {
    bool rc = true;
    out->set_broadcast(broadcast_spec, arg0, arg1);
    switch (arg0->get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_divide, i32, arg0, arg1, out, broadcast_spec, pythondiv);
        OPENVINO_TYPE_CASE(evaluate_divide, i64, arg0, arg1, out, broadcast_spec, pythondiv);
        OPENVINO_TYPE_CASE(evaluate_divide, u32, arg0, arg1, out, broadcast_spec, pythondiv);
        OPENVINO_TYPE_CASE(evaluate_divide, u64, arg0, arg1, out, broadcast_spec, pythondiv);
        OPENVINO_TYPE_CASE(evaluate_divide, f16, arg0, arg1, out, broadcast_spec, pythondiv);
        OPENVINO_TYPE_CASE(evaluate_divide, f32, arg0, arg1, out, broadcast_spec, pythondiv);
        OPENVINO_TYPE_CASE(evaluate_divide, bf16, arg0, arg1, out, broadcast_spec, pythondiv);
    default:
        rc = false;
        break;
    }
    return rc;
}

ov::Tensor equality_mask(const ov::Tensor& tensor, const shared_ptr<op::Constant>& constant) {
    auto mask_out = ov::TensorVector{{element::boolean, tensor.get_shape()}};

    auto c_tensor = ov::Tensor(constant->get_element_type(), constant->get_shape());
    memcpy(c_tensor.data(), constant->get_data_ptr(), c_tensor.get_byte_size());

    const auto& param = std::make_shared<op::Parameter>(tensor.get_element_type(), tensor.get_shape());
    op::v1::Equal(param, constant).evaluate(mask_out, ov::TensorVector{tensor, c_tensor});
    return mask_out.front();
}

ov::Tensor or_tensor(const ov::Tensor& lhs, const ov::Tensor& rhs) {
    auto logical_or = op::v1::LogicalOr(std::make_shared<op::Parameter>(lhs.get_element_type(), lhs.get_shape()),
                                        std::make_shared<op::Parameter>(rhs.get_element_type(), rhs.get_shape()),
                                        ngraph::op::AutoBroadcastType::NUMPY);

    auto outs = ov::TensorVector{{lhs.get_element_type(), logical_or.get_output_shape(0)}};
    logical_or.evaluate(outs, ov::TensorVector{lhs, rhs});
    return outs.front();
}

bool evaluate_bound(const Node* node, ov::TensorVector& output_values, bool is_upper) {
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

    auto input1_low = ov::evaluate_lower_bound(input1);
    if (!input1_low)
        return false;
    auto input1_up = ov::evaluate_upper_bound(input1);
    if (!input1_up)
        return false;
    auto input2_low = ov::evaluate_lower_bound(input2);
    if (!input2_low)
        return false;
    auto input2_up = ov::evaluate_upper_bound(input2);
    if (!input2_up)
        return false;

    auto zeros_const = op::Constant::create(input2.get_element_type(), {}, {0});
    const auto zero_t = ov::Tensor(input2.get_element_type(), Shape{});
    memcpy(zero_t.data(), zeros_const->get_data_ptr(), zero_t.get_byte_size());

    OPENVINO_SUPPRESS_DEPRECATED_START
    auto max_constant = get_constant_max_of_type(input2.get_element_type());
    auto dynamic_mask = or_tensor(equality_mask(input1_up, max_constant), equality_mask(input2_up, max_constant));
    OPENVINO_SUPPRESS_DEPRECATED_END

    // mask to find out positive values for arg2
    auto less_up_outputs = ov::TensorVector{{element::boolean, input2.get_shape()}};
    auto& input2_positive_up_mask = less_up_outputs.front();

    bool status = op::v1::Less().evaluate(less_up_outputs, ov::TensorVector{zero_t, input2_up});
    if (!status)
        return status;

    // mask to find out negative values for arg2
    auto less_low_outputs = ov::TensorVector{{element::boolean, input2.get_shape()}};
    auto& input2_negative_low_mask = less_low_outputs.front();
    status = op::v1::Less().evaluate(less_low_outputs, {input2_low, zero_t});
    if (!status)
        return status;

    // mask to find out ranges around 0 for arg2
    auto logical_and_up_outputs = ov::TensorVector{{element::boolean, input2.get_shape()}};
    auto& input2_low_negative_up_positive_mask = logical_and_up_outputs.front();
    status = op::v1::LogicalAnd().evaluate(logical_and_up_outputs, {input2_negative_low_mask, input2_positive_up_mask});
    if (!status)
        return status;

    auto value1_outs = ov::TensorVector{{input1.get_element_type(), input_shape.get_shape()}};
    auto& value1 = value1_outs.front();

    auto value2_outs = ov::TensorVector{{input2.get_element_type(), input2.get_shape()}};
    auto& value2 = value2_outs.front();

    if (!is_upper) {
        status = op::v1::Select().evaluate(value1_outs, {input2_positive_up_mask, input1_low, input1_up});
        if (!status)
            return status;

        status = op::v1::Select().evaluate(value2_outs, {input2_positive_up_mask, input2_up, input2_low});
        if (!status)
            return status;

        status = node->evaluate(output_values, ov::TensorVector{value1, value2});
        if (!status)
            return status;

        // replace values where zeros inside range of second arg to maximum values
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto output_minimum_value = get_constant_min_of_type(output_values[0].get_element_type());
        OPENVINO_SUPPRESS_DEPRECATED_END
        if (output_minimum_value == nullptr)
            return false;

        auto out_min_v = ov::Tensor(output_minimum_value->get_element_type(), output_minimum_value->get_shape());
        memcpy(out_min_v.data(), output_minimum_value->get_data_ptr(), out_min_v.get_byte_size());

        status = op::v1::Select().evaluate(output_values,
                                           {input2_low_negative_up_positive_mask, out_min_v, output_values[0]});
        if (!status)
            return status;

        status = op::v1::Select().evaluate(output_values, {dynamic_mask, zero_t, output_values[0]});
        if (!status)
            return status;
    } else {
        status = op::v1::Select().evaluate(value1_outs, {input2_positive_up_mask, input1_up, input1_low});
        if (!status)
            return status;

        status = op::v1::Select().evaluate(value2_outs, {input2_positive_up_mask, input2_low, input2_up});
        if (!status)
            return status;

        // create mask where zeros in the second argument are placed
        auto eq_zero_mask = ov::TensorVector{{element::boolean, input2.get_shape()}};
        auto& input2_zeros_mask = eq_zero_mask.front();
        bool status = op::v1::Equal().evaluate(eq_zero_mask, {value2, zero_t});
        if (!status)
            return status;

        // replace zeros by 1 values to get result of divide for other values of arguments
        auto ones = op::Constant::create(input2.get_element_type(), input2.get_shape(), {1});
        auto ones_t = ov::Tensor(ones->get_element_type(), ones->get_shape());
        memcpy(ones_t.data(), ones->get_data_ptr(), ones_t.get_byte_size());

        status = op::v1::Select().evaluate(value2_outs, {input2_zeros_mask, ones_t, value2});
        if (!status)
            return status;

        status = node->evaluate(output_values, {value1, value2});
        if (!status)
            return status;

        // replace values where zeros were found in the second argument to maximum values
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto output_maximum_value = get_constant_max_of_type(output_values[0].get_element_type());
        OPENVINO_SUPPRESS_DEPRECATED_END
        if (output_maximum_value == nullptr)
            return false;

        auto out_max_v = ov::Tensor(output_maximum_value->get_element_type(), output_maximum_value->get_shape());
        memcpy(out_max_v.data(), output_maximum_value->get_data_ptr(), out_max_v.get_byte_size());

        status = op::v1::Select().evaluate(output_values, {input2_zeros_mask, out_max_v, output_values[0]});
        if (!status)
            return status;

        // replace values where zeros inside [low, ip] values range of second arg to maximum values
        status = op::v1::Select().evaluate(output_values,
                                           {input2_low_negative_up_positive_mask, out_max_v, output_values[0]});
        if (!status)
            return status;

        // in case input elements were dynamic we replace them with zero
        status = op::v1::Select().evaluate(output_values, {dynamic_mask, out_max_v, output_values[0]});
        if (!status)
            return status;
    }
    return status;
}
}  // namespace
}  // namespace divide

// ------------------------------ v1 -------------------------------------------

op::v1::Divide::Divide(const Output<Node>& arg0, const Output<Node>& arg1, const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast) {
    constructor_validate_and_infer_types();
}

op::v1::Divide::Divide(const Output<Node>& arg0,
                       const Output<Node>& arg1,
                       bool pythondiv,
                       const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast),
      m_pythondiv(pythondiv) {
    constructor_validate_and_infer_types();
}

bool op::v1::Divide::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_Divide_visit_attributes);
    BinaryElementwiseArithmetic::visit_attributes(visitor);
    visitor.on_attribute("m_pythondiv", m_pythondiv);
    return true;
}

shared_ptr<Node> op::v1::Divide::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Divide_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v1::Divide>(new_args.at(0), new_args.at(1), this->is_pythondiv(), this->get_autob());
}

bool op::v1::Divide::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v1_Divide_evaluate);
    return divide::evaluate_divide(inputs[0], inputs[1], outputs[0], get_autob(), is_pythondiv());
}

bool op::v1::Divide::has_evaluate() const {
    OV_OP_SCOPE(v1_Divide_has_evaluate);
    switch (get_input_element_type(0)) {
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::bf16:
    case ngraph::element::f32:
        return true;
    default:
        break;
    }
    return false;
}

bool ov::op::v1::Divide::evaluate_lower(TensorVector& outputs) const {
    return divide::evaluate_bound(this, outputs, false);
}

bool ov::op::v1::Divide::evaluate_upper(TensorVector& outputs) const {
    return divide::evaluate_bound(this, outputs, true);
}
