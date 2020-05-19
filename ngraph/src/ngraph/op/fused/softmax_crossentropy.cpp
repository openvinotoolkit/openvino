//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include "ngraph/op/fused/softmax_crossentropy.hpp"

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::SoftmaxCrossEntropy::type_info;

op::SoftmaxCrossEntropy::SoftmaxCrossEntropy(const Output<Node>& arg1,
                                             const Output<Node>& arg2,
                                             bool soft_label,
                                             int64_t ignore_index)
    : FusedOp({arg1, arg2})
    , m_soft_label(soft_label)
    , m_ignore_index(ignore_index)
{
    constructor_validate_and_infer_types();
}

NodeVector op::SoftmaxCrossEntropy::decompose_op() const
{
    auto input_to_normalize = input_value(0);
    auto labels = input_value(1);
    auto reduction_axis = input_to_normalize.get_shape().size() - 1;

    auto create_mask = [&]() -> std::shared_ptr<ngraph::Node> {
        // ignore mask
        auto mask_constant = ngraph::op::Constant::create(
            labels.get_element_type(), labels.get_shape(), {m_ignore_index});
        auto not_equal = std::make_shared<ngraph::op::NotEqual>(labels, mask_constant);
        auto convert =
            std::make_shared<ngraph::op::Convert>(not_equal, input_to_normalize.get_element_type());
        auto reshape = std::make_shared<ngraph::op::Reshape>(
            convert, AxisVector{0, 1}, Shape{convert->get_shape().at(0), 1});
        return reshape;
    };

    auto create_xe = [&](std::shared_ptr<ngraph::Node> one_hot,
                         std::shared_ptr<ngraph::Node> input_softmax) {
        auto node_log = std::make_shared<ngraph::op::Log>(input_softmax);
        auto node_mul = one_hot * node_log;
        auto node_sum = std::make_shared<ngraph::op::Sum>(
            node_mul, AxisSet{static_cast<size_t>(reduction_axis)});
        return -node_sum;
    };

    if (m_soft_label)
    {
        // always reduces the sum on the last axis
        auto max_xj = std::make_shared<ngraph::op::Max>(
            input_to_normalize, AxisSet{static_cast<size_t>(reduction_axis)});
        auto broadcast_max_xj = std::make_shared<ngraph::op::Broadcast>(
            max_xj, input_to_normalize.get_shape(), AxisSet{1});
        auto subtract =
            std::make_shared<ngraph::op::Subtract>(input_to_normalize, broadcast_max_xj);
        auto exp = std::make_shared<ngraph::op::Exp>(subtract);

        auto sum_over_j =
            std::make_shared<ngraph::op::Sum>(exp, AxisSet{static_cast<size_t>(reduction_axis)});
        auto log_sum_over_j = std::make_shared<ngraph::op::Log>(sum_over_j);

        auto subtract_max_xj_from_input =
            std::make_shared<ngraph::op::Subtract>(input_to_normalize, broadcast_max_xj);
        auto broadcast_log = std::make_shared<ngraph::op::Broadcast>(
            log_sum_over_j, subtract_max_xj_from_input->get_shape(), AxisSet{1});
        auto subtract_max_xj_from_input_from_log_sum_over_j =
            std::make_shared<ngraph::op::Subtract>(subtract_max_xj_from_input, broadcast_log);

        // insert dtype conversion if required
        if (labels.get_element_type() != input_to_normalize.get_element_type())
        {
            labels = std::make_shared<ngraph::op::Convert>(labels,
                                                           input_to_normalize.get_element_type());
        }
        auto multiply = std::make_shared<ngraph::op::Multiply>(
            labels, subtract_max_xj_from_input_from_log_sum_over_j);
        auto sum_over_k = std::make_shared<ngraph::op::Sum>(
            multiply, AxisSet{static_cast<size_t>(reduction_axis)});
        auto negate_summation = std::make_shared<ngraph::op::Negative>(sum_over_k);
        auto reshape = std::make_shared<ngraph::op::Reshape>(
            negate_summation, AxisVector{0}, Shape{input_to_normalize.get_shape().at(0), 1});
        return {reshape};
    }
    else
    {
        // we will have one_hot encoding on labels if softmax_lables = false
        size_t one_hot_axis = input_to_normalize.get_shape().size() - 1;
        size_t softmax_axis = input_to_normalize.get_shape().size() - 1;
        auto reshape_labels =
            make_shared<op::Reshape>(labels, AxisVector{0, 1}, Shape{labels.get_shape().at(0)});
        auto one_hot_labels = std::make_shared<ngraph::op::OneHot>(
            reshape_labels, input_to_normalize.get_shape(), one_hot_axis);
        auto convert_one_hot = std::make_shared<ngraph::op::Convert>(
            one_hot_labels, input_to_normalize.get_element_type());
        auto mask = create_mask();
        // softmax will be applied on the input to cross_entropy
        auto softmax =
            std::make_shared<ngraph::op::Softmax>(input_to_normalize, AxisSet{softmax_axis});
        auto xe = create_xe(convert_one_hot, softmax);
        auto reshape_xe = std::make_shared<ngraph::op::Reshape>(
            xe, AxisVector{0}, Shape{xe->get_shape().at(0), 1});
        return {reshape_xe * mask};
    }
}

void op::SoftmaxCrossEntropy::pre_validate_and_infer_types()
{
    element::Type input_element_type = get_input_element_type(0);
    PartialShape data_pshape = get_input_partial_shape(0);
    PartialShape labels_pshape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");

    if (data_pshape.is_dynamic() || labels_pshape.is_dynamic())
    {
        set_output_type(0, input_element_type, PartialShape::dynamic());
    }
}

shared_ptr<Node> op::SoftmaxCrossEntropy::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<SoftmaxCrossEntropy>(
        new_args.at(0), new_args.at(1), m_soft_label, m_ignore_index);
}

constexpr NodeTypeInfo op::SoftmaxCrossEntropyBackprop::type_info;

op::SoftmaxCrossEntropyBackprop::SoftmaxCrossEntropyBackprop(const Output<Node>& delta,
                                                             const Output<Node>& softmax,
                                                             const Output<Node>& labels,
                                                             bool soft_label,
                                                             int64_t ignore_index)
    : FusedOp({delta, softmax, labels})
    , m_soft_label(soft_label)
    , m_ignore_index(ignore_index)
{
    constructor_validate_and_infer_types();
}

void op::SoftmaxCrossEntropyBackprop::pre_validate_and_infer_types()
{
    element::Type input_element_type = get_input_element_type(0);
    PartialShape delta_pshape = get_input_partial_shape(0);
    PartialShape softmax_pshape = get_input_partial_shape(1);
    PartialShape labels_pshape = get_input_partial_shape(2);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");

    if (delta_pshape.is_dynamic() || softmax_pshape.is_dynamic() || labels_pshape.is_dynamic())
    {
        set_output_type(0, input_element_type, PartialShape::dynamic());
    }
}

shared_ptr<Node>
    op::SoftmaxCrossEntropyBackprop::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<SoftmaxCrossEntropyBackprop>(
        new_args.at(0), new_args.at(1), new_args.at(2), m_soft_label, m_ignore_index);
}

NodeVector op::SoftmaxCrossEntropyBackprop::decompose_op() const
{
    auto delta = input_value(0);
    auto softmax = input_value(1);
    auto labels = input_value(2);
    size_t one_hot_axis = delta.get_shape().size() - 1;
    // always reduces the sum on the last axis
    auto reduction_axis = delta.get_shape().size() - 1;

    if (m_soft_label)
    {
        if (delta.get_shape() != labels.get_shape())
        {
            auto reshape = std::make_shared<ngraph::op::Reshape>(
                delta, AxisVector{0, 1}, Shape{delta.get_shape().at(0)});
            delta =
                std::make_shared<ngraph::op::Broadcast>(reshape, labels.get_shape(), AxisSet{1});
        }
        auto delta_mul_labels = std::make_shared<ngraph::op::Multiply>(delta, labels);
        auto summation_delta_mul_labels = std::make_shared<ngraph::op::Sum>(
            delta_mul_labels, AxisSet{static_cast<size_t>(reduction_axis)});
        auto broadcast_sum = std::make_shared<ngraph::op::Broadcast>(
            summation_delta_mul_labels, softmax.get_shape(), AxisSet{1});
        auto multiply_sm = broadcast_sum * softmax;
        return {multiply_sm - delta_mul_labels};
    }
    else
    {
        // ignore mask
        auto mask_constant =
            ngraph::op::Constant::create(element::i64, labels.get_shape(), {m_ignore_index});
        auto not_equal = std::make_shared<ngraph::op::NotEqual>(labels, mask_constant);
        auto convert = std::make_shared<ngraph::op::Convert>(not_equal, delta.get_element_type());
        auto reshape = std::make_shared<ngraph::op::Reshape>(
            convert, AxisVector{0, 1}, Shape{convert->get_shape().at(0)});
        auto broadcast_mask =
            std::make_shared<ngraph::op::Broadcast>(reshape, softmax.get_shape(), AxisSet{1});

        // one hot encoding of labels
        auto reshape_labels =
            make_shared<op::Reshape>(labels, AxisVector{0, 1}, Shape{labels.get_shape().at(0)});
        auto one_hot =
            std::make_shared<ngraph::op::OneHot>(reshape_labels, softmax.get_shape(), one_hot_axis);
        auto convert_one_hot =
            std::make_shared<ngraph::op::Convert>(one_hot, delta.get_element_type());

        if (delta.get_shape() != convert_one_hot->get_shape())
        {
            auto reshape = std::make_shared<ngraph::op::Reshape>(
                delta, AxisVector{0, 1}, Shape{delta.get_shape().at(0)});
            delta = std::make_shared<ngraph::op::Broadcast>(
                reshape, convert_one_hot->get_shape(), AxisSet{1});
        }

        // (cross_entr * delta * mask)
        auto delta_mul_labels = std::make_shared<ngraph::op::Multiply>(delta, convert_one_hot);
        auto multiply_mask =
            std::make_shared<ngraph::op::Multiply>(delta_mul_labels, broadcast_mask);

        // sum (cross_entr * delta * mask)
        auto summation_delta_mul_labels = std::make_shared<ngraph::op::Sum>(
            multiply_mask, AxisSet{static_cast<size_t>(reduction_axis)});

        auto broadcast_sum = std::make_shared<ngraph::op::Broadcast>(
            summation_delta_mul_labels, softmax.get_shape(), AxisSet{1});
        auto multiply_sm_with_summation = broadcast_sum * softmax;
        return {multiply_sm_with_summation - multiply_mask};
    }
}
