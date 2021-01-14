//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include <functional>
#include <memory>

#include "ngraph/builder/norm.hpp"
#include "ngraph/node.hpp"
#include "onnx_import/default_opset.hpp"
#include "onnx_import/exceptions.hpp"
#include "onnx_import/utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace
            {
                std::shared_ptr<default_opset::Constant> get_reduction_axes(const Node& node)
                {
                    auto reduction_axes =
                        node.get_attribute_value<std::vector<std::int64_t>>("axes", {});

                    const auto input_rank = node.get_ng_inputs().at(0).get_partial_shape().rank();

                    if (reduction_axes.empty())
                    {
                        NGRAPH_CHECK(input_rank.is_static(),
                                     "The input tensor's rank needs to be known(static) when the "
                                     "'axes' attribute is not specified. Node: ",
                                     node.get_description());

                        reduction_axes = onnx_import::common::get_monotonic_range<int64_t>(
                            input_rank.get_length());
                    }

                    if (input_rank.is_static())
                    {
                        CHECK_VALID_NODE(node,
                                         reduction_axes.size() <= input_rank.get_length(),
                                         "Number of reduction axes (",
                                         reduction_axes.size(),
                                         ") is larger than the input tensor's rank (",
                                         input_rank.get_length(),
                                         ")");
                    }

                    return default_opset::Constant::create(
                        element::i64, Shape{reduction_axes.size()}, reduction_axes);
                }

                template <typename OpType>
                std::shared_ptr<ngraph::Node>
                    make_ng_reduction_op(const Node& node, const Output<ngraph::Node>& ng_input)
                {
                    const auto reduction_axes = get_reduction_axes(node);
                    const std::int64_t keepdims =
                        node.get_attribute_value<std::int64_t>("keepdims", 1);

                    const auto op_node = std::make_shared<OpType>(
                        ng_input, reduction_axes, static_cast<bool>(keepdims));

                    return op_node;
                }
            } // namespace

            namespace set_1
            {
                OutputVector reduce_log_sum(const Node& node)
                {
                    const Output<ngraph::Node> sum_node =
                        make_ng_reduction_op<default_opset::ReduceSum>(node,
                                                                       node.get_ng_inputs().at(0));
                    return {std::make_shared<default_opset::Log>(sum_node)};
                }

                OutputVector reduce_log_sum_exp(const Node& node)
                {
                    const auto exp_node =
                        std::make_shared<default_opset::Exp>(node.get_ng_inputs().at(0));
                    const Output<ngraph::Node> sum_node =
                        make_ng_reduction_op<default_opset::ReduceSum>(node, exp_node);
                    return {std::make_shared<default_opset::Log>(sum_node)};
                }

                OutputVector reduce_l1(const Node& node)
                {
                    return {make_ng_reduction_op<default_opset::ReduceL1>(
                        node, node.get_ng_inputs().at(0))};
                }

                OutputVector reduce_l2(const Node& node)
                {
                    return {make_ng_reduction_op<default_opset::ReduceL2>(
                        node, node.get_ng_inputs().at(0))};
                }

                OutputVector reduce_max(const Node& node)
                {
                    return {make_ng_reduction_op<default_opset::ReduceMax>(
                        node, node.get_ng_inputs().at(0))};
                }

                OutputVector reduce_mean(const Node& node)
                {
                    return {make_ng_reduction_op<default_opset::ReduceMean>(
                        node, node.get_ng_inputs().at(0))};
                }

                OutputVector reduce_min(const Node& node)
                {
                    return {make_ng_reduction_op<default_opset::ReduceMin>(
                        node, node.get_ng_inputs().at(0))};
                }

                OutputVector reduce_prod(const Node& node)
                {
                    return {make_ng_reduction_op<default_opset::ReduceProd>(
                        node, node.get_ng_inputs().at(0))};
                }

                OutputVector reduce_sum(const Node& node)
                {
                    return {make_ng_reduction_op<default_opset::ReduceSum>(
                        node, node.get_ng_inputs().at(0))};
                }

                OutputVector reduce_sum_square(const Node& node)
                {
                    const auto input = Output<ngraph::Node>{node.get_ng_inputs().at(0)};
                    const auto square_node =
                        std::make_shared<default_opset::Multiply>(input, input);
                    return {make_ng_reduction_op<default_opset::ReduceSum>(node, square_node)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
