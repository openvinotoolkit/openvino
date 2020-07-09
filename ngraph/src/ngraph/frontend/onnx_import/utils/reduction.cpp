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

#include <cstddef> // std::size_t
#include <vector>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/op/constant.hpp"
#include "reduction.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace reduction
        {
            namespace detail
            {
                AxisSet get_reduction_axes(const Node& node)
                {
                    auto reduction_axes =
                        node.get_attribute_value<std::vector<std::int64_t>>("axes", {});

                    const auto input_rank = node.get_ng_inputs().at(0).get_partial_shape().rank();

                    std::vector<std::size_t> normalized_axes =
                        ngraph::normalize_axes(node.get_description(), reduction_axes, input_rank);

                    if (reduction_axes.empty())
                    {
                        NGRAPH_CHECK(input_rank.is_static(),
                                     "The input tensor's rank needs to be known(static) when the "
                                     "'axes' attribute is not specified. Node: ",
                                     node.get_description());

                        normalized_axes = onnx_import::common::get_monotonic_range<size_t>(
                            input_rank.get_length());
                    }
                    return AxisSet{normalized_axes};
                }
            } // namespace  detail

            std::shared_ptr<ngraph::Node> make_ng_reduction_op(const Node& node,
                                                               const Output<ngraph::Node>& ng_input,
                                                               ReductionFunction reduction_function)
            {
                auto data_shape = ng_input.get_shape();

                auto reduction_axes = detail::get_reduction_axes(node);

                ASSERT_VALID_ARGUMENT(node, reduction_axes.size() <= data_shape.size())
                    << "provided reduction axes count (" << reduction_axes.size()
                    << ") is larger than input tensor rank (" << data_shape.size() << ")";

                std::shared_ptr<ngraph::Node> op_node =
                    reduction_function(ng_input, reduction_axes);

                std::int64_t keepdims = node.get_attribute_value<std::int64_t>("keepdims", 1);
                if (keepdims == 0)
                {
                    return op_node;
                }

                auto output_shape = data_shape;
                // flatten reduced axes and preserve original dimensions count.
                for (const auto& idx : reduction_axes)
                {
                    output_shape.at(idx) = 1;
                }
                return builder::opset1::reshape(op_node, output_shape);
            }

            std::shared_ptr<ngraph::Node>
                make_ng_reduction_op(const Node& node,
                                     const Output<ngraph::Node>& ng_input,
                                     RuntimeReductionFunction reduction_function)
            {
                const auto data_ps = node.get_ng_inputs().at(0).get_partial_shape();
                NGRAPH_CHECK(data_ps.rank().is_static(),
                             "Reduction operations input rank is required to be static");

                const auto data_rank = data_ps.rank().get_length();

                const auto reduction_axes = detail::get_reduction_axes(node);

                ASSERT_VALID_ARGUMENT(node, reduction_axes.size() <= data_rank)
                    << "provided reduction axes count (" << reduction_axes.size()
                    << ") is larger than input tensor rank (" << data_rank << ")";

                std::int64_t keepdims = node.get_attribute_value<std::int64_t>("keepdims", 1);

                const auto op_node = reduction_function(
                    ng_input,
                    std::make_shared<default_opset::Constant>(element::i64,
                                                              ngraph::Shape{reduction_axes.size()},
                                                              reduction_axes.to_vector()),
                    static_cast<bool>(keepdims));

                return op_node;
            }

        } // namespace  reduction
    }     // namespace onnx_import
} // namespace ngraph
