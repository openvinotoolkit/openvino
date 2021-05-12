// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/builder/norm.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/reduce_sum.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/shape.hpp"

using namespace std;

namespace ngraph
{
    namespace builder
    {
        namespace detail
        {
            namespace opset1
            {
                shared_ptr<Node> lp_norm(const Output<Node>& value,
                                         size_t p_norm,
                                         const Output<Node>& reduction_axes,
                                         float bias)
                {
                    // In general "entrywise" lp-norm for matrix `A` is defined as following double
                    // sum:
                    // ||A||_p = ||vec(A)||_p = [sum_{i=1}^m sum_{j=1}^n abs(a_{i,j})^p]^{1/p}
                    shared_ptr<Node> abs_values{make_shared<ngraph::opset1::Abs>(value)};
                    shared_ptr<Node> p_node = ngraph::opset1::Constant::create(
                        value.get_element_type(), Shape{}, {p_norm});

                    // Get inner part of equation: abs_values^p_node, then sum over reduction_axes.
                    shared_ptr<Node> values{make_shared<ngraph::opset1::Power>(abs_values, p_node)};
                    values = make_shared<ngraph::opset1::ReduceSum>(values, reduction_axes, false);

                    shared_ptr<Node> bias_node{ngraph::opset1::Constant::create(
                        values->get_element_type(), Shape{}, {bias})};

                    values = make_shared<ngraph::opset1::Add>(values, bias_node);

                    // Get outer part of equation: raise values to 1/p_norm exponent.
                    shared_ptr<Node> inv_p_node = ngraph::opset1::Constant::create(
                        values->get_element_type(), Shape{}, {1.f / p_norm});

                    return {make_shared<ngraph::opset1::Power>(values, inv_p_node)
                                ->add_provenance_group_members_above({value})};
                }
            } // namespace opset1
        }     // namespace detail

        shared_ptr<Node> builder::opset1::l0_norm(const Output<Node>& value,
                                                  const Output<Node>& reduction_axes)
        {
            // L0 norm returns number of elements different from zero.
            const shared_ptr<Node> zero_node{
                ngraph::opset1::Constant::create(value.get_element_type(), Shape{}, {0.f})};

            // Convert bool values to input node data type.
            const shared_ptr<Node> non_zero_values = make_shared<ngraph::opset1::Convert>(
                make_shared<ngraph::opset1::NotEqual>(value, zero_node), value.get_element_type());

            return make_shared<ngraph::opset1::ReduceSum>(non_zero_values, reduction_axes, false)
                ->add_provenance_group_members_above({value});
        }

        shared_ptr<Node> builder::opset1::l1_norm(const Output<Node>& value,
                                                  const Output<Node>& reduction_axes,
                                                  float bias)
        {
            const shared_ptr<Node> values{make_shared<ngraph::opset1::ReduceSum>(
                make_shared<ngraph::opset1::Abs>(value), reduction_axes, false)};

            const shared_ptr<Node> bias_node{
                ngraph::opset1::Constant::create(values->get_element_type(), Shape{}, {bias})};

            return make_shared<ngraph::opset1::Add>(values, bias_node)
                ->add_provenance_group_members_above({value});
        }

        shared_ptr<Node> builder::opset1::l2_norm(const Output<Node>& value,
                                                  const Output<Node>& reduction_axes,
                                                  float bias,
                                                  BiasMode bias_mode,
                                                  bool keep_dims)
        {
            shared_ptr<Node> values{make_shared<ngraph::opset1::ReduceSum>(
                make_shared<ngraph::opset1::Multiply>(value, value), reduction_axes, keep_dims)};

            shared_ptr<Node> bias_node{
                ngraph::opset1::Constant::create(values->get_element_type(), Shape{}, {bias})};
            shared_ptr<Node> result;
            switch (bias_mode)
            {
            case BiasMode::MAX:
            {
                result = make_shared<ngraph::opset1::Sqrt>(
                    make_shared<ngraph::opset1::Maximum>(values, bias_node));
                break;
            }
            case BiasMode::ADD:
            default:
                result = make_shared<ngraph::opset1::Sqrt>(
                    make_shared<ngraph::opset1::Add>(values, bias_node));
            }
            return result->add_provenance_group_members_above({value});
        }

        shared_ptr<Node> builder::opset1::lp_norm(const Output<Node>& value,
                                                  const Output<Node>& reduction_axes,
                                                  size_t p_norm,
                                                  float bias)
        {
            // The number of non-zero elements
            if (p_norm == 0)
            {
                return opset1::l0_norm(value, reduction_axes);
            }
            //  sum of absolute values.
            else if (p_norm == 1)
            {
                return opset1::l1_norm(value, reduction_axes, bias);
            }
            // sqrt of sum of squares - Euclidean norm
            else if (p_norm == 2)
            {
                return opset1::l2_norm(value, reduction_axes, bias);
            }
            // generic case
            else
            {
                return detail::opset1::lp_norm(value, p_norm, reduction_axes, bias);
            }
        }

    } // namespace builder

} // namespace ngraph
