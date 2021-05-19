// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/add.hpp"
#include "default_opset.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector add(const Node& node)
                {
                    const Output<ngraph::Node> lhs_node = node.get_ng_inputs().at(0);
                    Output<ngraph::Node> rhs_node = node.get_ng_inputs().at(1);
                    const bool broadcast = node.get_attribute_value<std::int64_t>("broadcast", 0);
                    if (broadcast)
                    {
                        if (node.has_attribute("axis"))
                        {
                            NGRAPH_CHECK((lhs_node.get_partial_shape().rank().is_static() &&
                                          rhs_node.get_partial_shape().rank().is_static()),
                                         "Add input's rank has to be static.");
                            // Unidirectional broadcast right node to left shape.
                            auto axis = node.get_attribute_value<std::int64_t>("axis");
                            auto lhs_rank = lhs_node.get_partial_shape().rank().get_length();
                            auto rhs_rank = rhs_node.get_partial_shape().rank().get_length();
                            if (axis < 0)
                                axis += lhs_rank;
                            if (lhs_rank > axis + rhs_rank)
                            {
                                auto ones = default_opset::Constant::create(
                                    element::i64,
                                    Shape{static_cast<size_t>(lhs_rank - axis - rhs_rank)},
                                    std::vector<int64_t>(lhs_rank - axis - rhs_rank, 1));
                                auto rhs_shape = std::make_shared<default_opset::ShapeOf>(rhs_node);
                                auto new_shape = std::make_shared<default_opset::Concat>(
                                    OutputVector{rhs_shape, ones}, 0);
                                rhs_node = std::make_shared<default_opset::Reshape>(
                                    rhs_node, new_shape, false);
                            }
                        }
                        else
                        {
                            rhs_node = std::make_shared<default_opset::Broadcast>(
                                rhs_node, std::make_shared<default_opset::ShapeOf>(lhs_node));
                        }
                    }
                    return {std::make_shared<default_opset::Add>(lhs_node, rhs_node)};
                }

            } // namespace set_1

            namespace set_7
            {
                OutputVector add(const Node& node)
                {
                    return {std::make_shared<default_opset::Add>(node.get_ng_inputs().at(0),
                                                                 node.get_ng_inputs().at(1))};
                }

            } // namespace set_7

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
