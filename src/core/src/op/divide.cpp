// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/divide.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/reference/divide.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

namespace divide {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorPtr& arg0,
              const HostTensorPtr& arg1,
              const HostTensorPtr& out,
              const op::AutoBroadcastSpec& broadcast_spec,
              bool pythondiv) {
    runtime::reference::divide(arg0->get_data_ptr<ET>(),
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
        NGRAPH_TYPE_CASE(evaluate_divide, i32, arg0, arg1, out, broadcast_spec, pythondiv);
        NGRAPH_TYPE_CASE(evaluate_divide, i64, arg0, arg1, out, broadcast_spec, pythondiv);
        NGRAPH_TYPE_CASE(evaluate_divide, u32, arg0, arg1, out, broadcast_spec, pythondiv);
        NGRAPH_TYPE_CASE(evaluate_divide, u64, arg0, arg1, out, broadcast_spec, pythondiv);
        NGRAPH_TYPE_CASE(evaluate_divide, f16, arg0, arg1, out, broadcast_spec, pythondiv);
        NGRAPH_TYPE_CASE(evaluate_divide, f32, arg0, arg1, out, broadcast_spec, pythondiv);
        NGRAPH_TYPE_CASE(evaluate_divide, bf16, arg0, arg1, out, broadcast_spec, pythondiv);
    default:
        rc = false;
        break;
    }
    return rc;
}

HostTensorPtr equality_mask(const HostTensorPtr& tensor, const shared_ptr<op::Constant>& constant) {
    auto mask = std::make_shared<HostTensor>(element::boolean, tensor->get_shape());
    const auto& param = std::make_shared<op::Parameter>(tensor->get_element_type(), tensor->get_shape());
    op::v1::Equal(param, constant, ngraph::op::AutoBroadcastType::NUMPY)
        .evaluate({mask}, {tensor, std::make_shared<HostTensor>(constant)});
    return mask;
}

HostTensorPtr or_tensor(const HostTensorPtr& lhs, const HostTensorPtr& rhs) {
    auto result = std::make_shared<HostTensor>();
    op::v1::LogicalOr(std::make_shared<op::Parameter>(lhs->get_element_type(), lhs->get_shape()),
                      std::make_shared<op::Parameter>(rhs->get_element_type(), rhs->get_shape()),
                      ngraph::op::AutoBroadcastType::NUMPY)
        .evaluate({result}, {lhs, rhs});
    return result;
}

bool evaluate_bound(const Node* node, const HostTensorVector& output_values, bool is_upper) {
    // for positive arg2 divide will have limits [low/up , up/low]
    // for negative arg2 limits for divide will be [up/low, low/up]
    // for arg2 range with both positive and negative values, divide can give any result [-inf, inf]
    NGRAPH_CHECK(node, validate_host_tensor_vector(output_values, 1));
    const auto& input1 = node->input_value(0);
    const auto& input2 = node->input_value(1);

    // broadcast shapes to allocate tensors of correct size for operations with both inputs
    PartialShape input_shape = input1.get_partial_shape();
    NGRAPH_CHECK(PartialShape::broadcast_merge_into(input_shape, input2.get_partial_shape(), node->get_autob()),
                 "Argument shapes in divide operation are inconsistent.");

    const auto& input2_low = input2.get_tensor().get_lower_value();
    if (input2_low == nullptr)
        return false;
    const auto& input2_up = input2.get_tensor().get_upper_value();
    if (input2_up == nullptr)
        return false;
    const auto& input1_low = input1.get_tensor().get_lower_value();
    if (input1_low == nullptr)
        return false;
    const auto& input1_up = input1.get_tensor().get_upper_value();
    if (input1_up == nullptr)
        return false;

    auto zeros_const = op::Constant::create(input2.get_element_type(), {}, {0});
    auto max_constant = get_constant_max_of_type(input2.get_element_type());
    auto dynamic_mask = or_tensor(equality_mask(input1_up, max_constant), equality_mask(input2_up, max_constant));

    // mask to find out positive values for arg2
    auto input2_positive_up_mask = std::make_shared<HostTensor>(element::boolean, input2.get_shape());
    // mask to find out ranges around 0 for arg2
    auto input2_low_negative_up_positive_mask = std::make_shared<HostTensor>(element::boolean, input2.get_shape());

    bool status =
        op::v1::Less().evaluate({input2_positive_up_mask}, {std::make_shared<HostTensor>(zeros_const), input2_up});
    if (!status)
        return status;

    // mask to find out negative values for arg2
    auto input2_negative_low_mask = std::make_shared<HostTensor>(element::boolean, input2.get_shape());
    status =
        op::v1::Less().evaluate({input2_negative_low_mask}, {input2_low, std::make_shared<HostTensor>(zeros_const)});
    if (!status)
        return status;
    status = op::v1::LogicalAnd().evaluate({input2_low_negative_up_positive_mask},
                                           {input2_negative_low_mask, input2_positive_up_mask});
    if (!status)
        return status;

    if (!is_upper) {
        auto value1 = std::make_shared<HostTensor>(input1.get_element_type(), input_shape);
        status = op::v1::Select().evaluate({value1}, {input2_positive_up_mask, input1_low, input1_up});
        if (!status)
            return status;

        auto value2 = std::make_shared<HostTensor>(input2.get_element_type(), input2.get_shape());
        status = op::v1::Select().evaluate({value2}, {input2_positive_up_mask, input2_up, input2_low});
        if (!status)
            return status;

        OPENVINO_SUPPRESS_DEPRECATED_START
        status = node->evaluate(output_values, {value1, value2});
        OPENVINO_SUPPRESS_DEPRECATED_END
        if (!status)
            return status;

        // replace values where zeros inside range of second arg to maximum values
        auto output_minimum_value = get_constant_min_of_type(output_values[0]->get_element_type());
        if (output_minimum_value == nullptr)
            return false;
        status = op::v1::Select().evaluate(output_values,
                                           {input2_low_negative_up_positive_mask,
                                            std::make_shared<HostTensor>(output_minimum_value),
                                            output_values[0]});
        if (!status)
            return status;

        status = op::v1::Select().evaluate(output_values,
                                           {dynamic_mask, std::make_shared<HostTensor>(zeros_const), output_values[0]});
        if (!status)
            return status;
    } else {
        auto value1 = std::make_shared<HostTensor>(input1.get_element_type(), input_shape);
        status = op::v1::Select().evaluate({value1}, {input2_positive_up_mask, input1_up, input1_low});
        if (!status)
            return status;

        auto value2 = std::make_shared<HostTensor>(input2.get_element_type(), input2.get_shape());
        status = op::v1::Select().evaluate({value2}, {input2_positive_up_mask, input2_low, input2_up});
        if (!status)
            return status;

        // create mask where zeros in the second argument are placed
        auto input2_zeros_mask = std::make_shared<HostTensor>(element::boolean, input2.get_shape());
        bool status =
            op::v1::Equal().evaluate({input2_zeros_mask}, {value2, std::make_shared<HostTensor>(zeros_const)});
        if (!status)
            return status;

        // replace zeros by 1 values to get result of divide for other values of arguments
        auto ones = op::Constant::create(input2.get_element_type(), input2.get_shape(), {1});
        status = op::v1::Select().evaluate({value2}, {input2_zeros_mask, std::make_shared<HostTensor>(ones), value2});
        if (!status)
            return status;

        OPENVINO_SUPPRESS_DEPRECATED_START
        status = node->evaluate(output_values, {value1, value2});
        OPENVINO_SUPPRESS_DEPRECATED_END
        if (!status)
            return status;

        // replace values where zeros were found in the second argument to maximum values
        auto output_maximum_value = get_constant_max_of_type(output_values[0]->get_element_type());
        if (output_maximum_value == nullptr)
            return false;
        status = op::v1::Select().evaluate(
            output_values,
            {input2_zeros_mask, std::make_shared<HostTensor>(output_maximum_value), output_values[0]});
        if (!status)
            return status;

        // replace values where zeros inside [low, ip] values range of second arg to maximum values
        status = op::v1::Select().evaluate(output_values,
                                           {input2_low_negative_up_positive_mask,
                                            std::make_shared<HostTensor>(output_maximum_value),
                                            output_values[0]});
        if (!status)
            return status;

        // in case input elements were dynamic we replace them with zero
        status = op::v1::Select().evaluate(
            output_values,
            {dynamic_mask, std::make_shared<HostTensor>(output_maximum_value), output_values[0]});
        if (!status)
            return status;
    }
    return status;
}
}  // namespace
}  // namespace divide

// ------------------------------ v1 -------------------------------------------

BWDCMP_RTTI_DEFINITION(op::v1::Divide);

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

bool ov::op::v1::Divide::evaluate_lower(const HostTensorVector& outputs) const {
    return divide::evaluate_bound(this, outputs, false);
}

bool ov::op::v1::Divide::evaluate_upper(const HostTensorVector& outputs) const {
    return divide::evaluate_bound(this, outputs, true);
}
