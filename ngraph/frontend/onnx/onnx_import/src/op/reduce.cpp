// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/builder/norm.hpp"
#include "ngraph/node.hpp"
#include "op/identity.hpp"
#include "utils/common.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace
            {
                std::shared_ptr<ngraph::Node> get_dynamic_all_axes_range(const Node& node)
                {
                    const auto input = node.get_ng_inputs().at(0);
                    const auto shape_of_input = std::make_shared<default_opset::ShapeOf>(input);
                    const auto scalar =
                        default_opset::Constant::create(element::i32, Shape{1}, {0});
                    const auto rank_of_input =
                        std::make_shared<default_opset::ShapeOf>(shape_of_input);
                    const auto rank_of_input_scalar =
                        std::make_shared<default_opset::Squeeze>(rank_of_input, scalar);
                    const auto start = default_opset::Constant::create(element::i32, Shape{}, {0});
                    const auto step = default_opset::Constant::create(element::i32, Shape{}, {1});
                    return std::make_shared<default_opset::Range>(
                        start, rank_of_input_scalar, step, element::i64);
                }

                std::shared_ptr<ngraph::Node> get_reduction_axes_from_input(const Node& node)
                {
                    const std::int64_t noop_with_empty_axes =
                        node.get_attribute_value<std::int64_t>("noop_with_empty_axes", 0);
                    const auto input = node.get_ng_inputs().at(0);
                    if (node.get_ng_inputs().size() > 1)
                    {
                        const auto reduction_axes = node.get_ng_inputs().at(1);
                        const auto reduction_axes_rank = reduction_axes.get_partial_shape().rank();
                        NGRAPH_CHECK(reduction_axes.get_partial_shape().is_static(),
                                     "The axes tensor's shape needs to be known(static). Node: ",
                                     node.get_description());

                        if (reduction_axes_rank.get_length() != 0 &&
                            reduction_axes.get_shape() != Shape{0})
                        {
                            return reduction_axes.get_node_shared_ptr();
                        }
                    }

                    if (noop_with_empty_axes)
                    {
                        return nullptr;
                    }
                    else
                    {
                        return get_dynamic_all_axes_range(node);
                    }
                }

                std::shared_ptr<ngraph::Node> get_reduction_axes_from_attr(const Node& node)
                {
                    auto reduction_axes =
                        node.get_attribute_value<std::vector<std::int64_t>>("axes", {});

                    const auto input_rank = node.get_ng_inputs().at(0).get_partial_shape().rank();

                    if (reduction_axes.empty())
                    {
                        if (input_rank.is_static())
                        {
                            reduction_axes = onnx_import::common::get_monotonic_range<int64_t>(
                                input_rank.get_length());
                        }
                        else
                        {
                            return get_dynamic_all_axes_range(node);
                        }
                    }

                    if (input_rank.is_static())
                    {
                        CHECK_VALID_NODE(node,
                                         static_cast<int64_t>(reduction_axes.size()) <=
                                             input_rank.get_length(),
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
                    make_ng_reduction_op(const Node& node,
                                         const Output<ngraph::Node>& ng_input,
                                         bool axes_as_attr = true)
                {
                    const std::int64_t keepdims =
                        node.get_attribute_value<std::int64_t>("keepdims", 1);

                    const auto reduction_axes = axes_as_attr ? get_reduction_axes_from_attr(node)
                                                             : get_reduction_axes_from_input(node);
                    if (reduction_axes != nullptr)
                    {
                        return std::make_shared<OpType>(
                            ng_input, reduction_axes, static_cast<bool>(keepdims));
                    }
                    else
                    {
                        return op::set_1::identity(node).at(0).get_node_shared_ptr();
                    }
                }
            } // namespace

            namespace set_13
            {
                OutputVector reduce_sum(const Node& node)
                {
                    return {make_ng_reduction_op<default_opset::ReduceSum>(
                        node, node.get_ng_inputs().at(0), false)};
                }
            } // namespace set_13

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
