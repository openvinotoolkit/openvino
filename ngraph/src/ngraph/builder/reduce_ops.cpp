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

#include <numeric>

#include "ngraph/axis_set.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/reduce_ops.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace builder
    {
        size_t get_num_elements(const Shape& shape, const AxisSet& reduction_axes)
        {
            size_t N = 1;
            for (auto a : reduction_axes)
            {
                N *= shape[a];
            }
            return N;
        }

        std::shared_ptr<Node> get_num_elements(const Output<Node>& value,
                                               const Output<Node>& reduction_axes)
        {
            const auto value_shape = std::make_shared<ngraph::opset1::ShapeOf>(value);
            const auto dim_values = std::make_shared<ngraph::opset1::Gather>(
                value_shape,
                reduction_axes,
                ngraph::opset1::Constant::create(element::i64, {}, {0}));

            return std::make_shared<ngraph::opset1::ReduceProd>(
                dim_values, ngraph::opset1::Constant::create(element::i64, {}, {0}));
        }

        std::shared_ptr<Node> l2_norm(const Output<Node>& node, const AxisSet& reduction_axes)
        {
            auto x2 = node * node;
            auto x2sum = std::make_shared<op::Sum>(x2, reduction_axes);

            return std::make_shared<op::Sqrt>(x2sum)->add_provenance_group_members_above({node});
        }

        std::shared_ptr<Node> mean(const Output<Node>& value, const AxisSet& reduction_axes)
        {
            auto xsum = std::make_shared<op::Sum>(value, reduction_axes);

            auto N = get_num_elements(value.get_shape(), reduction_axes);
            const auto& et = value.get_element_type();

            auto divisor = op::Constant::create(et, xsum->get_shape(), {N});

            return (xsum / divisor)->add_provenance_group_members_above({value});
        }

        std::shared_ptr<Node> std_dev(const Output<Node>& node,
                                      const AxisSet& reduction_axes,
                                      const bool bessel_correction)
        {
            return std::make_shared<op::Sqrt>(variance(node, reduction_axes, bessel_correction))
                ->add_provenance_group_members_above({node});
        }

        // This currently calculates [E[X^2] - E[X]^2] instead of [E[(X-\mu)^2]]
        // The second might be more numerically stable/easier to pattern match
        // It also requires adding a broadcast op, and would probably be slower
        // TODO(mbrookhart): Switch to E[(X-\mu)^2]?
        std::shared_ptr<Node> variance(const Output<Node>& value,
                                       const AxisSet& reduction_axes,
                                       const bool bessel_correction)
        {
            std::shared_ptr<Node> mu = mean(value, reduction_axes);

            auto reshape = value.get_shape();
            for (auto i : reduction_axes)
            {
                reshape[i] = 1;
            }

            ngraph::AxisVector order = ngraph::get_default_order(mu->get_shape());

            mu = std::make_shared<op::Reshape>(mu, order, reshape);

            Output<Node> diff = make_with_numpy_broadcast<op::Subtract>(value, mu);

            diff = std::make_shared<op::Sum>(diff * diff, reduction_axes);

            const auto& et = value.get_element_type();
            auto N = get_num_elements(value.get_shape(), reduction_axes);

            std::shared_ptr<Node> result;
            if (bessel_correction)
            {
                auto N1const = op::Constant::create(et, diff.get_shape(), {N - 1});
                result = diff / N1const;
            }
            else
            {
                auto Nconst = op::Constant::create(et, diff.get_shape(), {N});
                result = diff / Nconst;
            }
            return result->add_provenance_group_members_above({value});
        }

        std::shared_ptr<Node> builder::opset1::mean(const Output<Node>& value,
                                                    const AxisSet& reduction_axes,
                                                    bool keep_dims)
        {
            std::shared_ptr<Node> elems_number;
            const auto value_elem_type = value.get_element_type();
            const auto reduction_axes_const = ngraph::opset1::Constant::create(
                element::i64, Shape{reduction_axes.size()}, reduction_axes.to_vector());
            const auto value_elems_sum =
                std::make_shared<ngraph::opset1::ReduceSum>(value, reduction_axes_const, keep_dims);
            if (value.get_partial_shape().is_static())
            {
                const auto elems_number_value = get_num_elements(value.get_shape(), reduction_axes);
                elems_number = ngraph::opset1::Constant::create(
                    value_elem_type, Shape{}, {elems_number_value});
            }
            else
            {
                elems_number = get_num_elements(value, reduction_axes_const);
                elems_number =
                    std::make_shared<ngraph::opset1::Convert>(elems_number, value_elem_type);
            }

            return std::make_shared<ngraph::opset1::Divide>(value_elems_sum, elems_number)
                ->add_provenance_group_members_above({value});
        }

        std::shared_ptr<Node> builder::opset1::variance(const Output<Node>& value,
                                                        const AxisSet& reduction_axes,
                                                        const bool bessel_correction)
        {
            const bool keep_dims = true;
            std::shared_ptr<Node> mu = opset1::mean(value, reduction_axes, keep_dims);

            Output<Node> diff = std::make_shared<ngraph::opset1::Subtract>(value, mu);

            diff = std::make_shared<ngraph::opset1::ReduceSum>(
                std::make_shared<ngraph::opset1::Multiply>(diff, diff),
                ngraph::opset1::Constant::create(
                    element::i64, Shape{reduction_axes.size()}, reduction_axes.to_vector()),
                false);

            const auto& et = value.get_element_type();
            const auto N = get_num_elements(value.get_shape(), reduction_axes);

            std::shared_ptr<Node> result;
            if (bessel_correction)
            {
                const auto N1const = ngraph::opset1::Constant::create(et, Shape{}, {N - 1});
                result = std::make_shared<ngraph::opset1::Divide>(diff, N1const);
            }
            else
            {
                const auto Nconst = ngraph::opset1::Constant::create(et, Shape{}, {N});
                result = std::make_shared<ngraph::opset1::Divide>(diff, Nconst);
            }
            return result->add_provenance_group_members_above({value});
        }

    } // namespace builder
} // namespace ngraph
