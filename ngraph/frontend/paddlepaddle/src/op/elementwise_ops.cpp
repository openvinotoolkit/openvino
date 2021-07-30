// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>

#include <ngraph/opsets/opset6.hpp>
#include <node_context.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                template <typename T>
                NamedOutputs elementwise_ops(const NodeContext& node)
                {
                    auto x = node.get_ng_input("X");
                    auto y = node.get_ng_input("Y");

                    auto axis = node.get_attribute<int>("axis");

                    PDPD_OP_VALIDATION_CHECK(node,
                                             x.get_partial_shape().rank().is_static(),
                                             "elementwise_ops: X rank must be static!");
                    PDPD_OP_VALIDATION_CHECK(node,
                                             y.get_partial_shape().rank().is_static(),
                                             "elementwise_ops: Y rank must be static!");
                    int64_t x_rank = x.get_partial_shape().rank().get_length();
                    int64_t y_rank = y.get_partial_shape().rank().get_length();

                    if ((axis == -1) || (axis == x_rank - 1) || (x_rank == y_rank))
                    {
                        return node.default_single_output_mapping({std::make_shared<T>(x, y)},
                                                                  {"Out"});
                    }
                    else
                    {
                        // This broadcast can be implemented by either ngraph::Reshape or
                        // ngraph::Broadcast. Since PDPD implicates y_shape is a subsequence of
                        // x_shape starting from axis, to use ngraph::Reshape like Paddle2ONNX,
                        // which is more friendly to PnP.
                        auto broadcast_shape = std::vector<int64_t>(x_rank, 1);
                        PartialShape y_shape = y.get_partial_shape();
                        int32_t i = 0;
                        for (auto it = y_shape.begin(); it != y_shape.end(); ++i, ++it)
                            broadcast_shape[axis + i] = (*it).get_length();

                        auto reshape_node =
                            ngraph::opset6::Constant::create(ngraph::element::i64,
                                                             ngraph::Shape{broadcast_shape.size()},
                                                             broadcast_shape);
                        auto y_node =
                            std::make_shared<ngraph::opset6::Reshape>(y, reshape_node, false);
                        return node.default_single_output_mapping({std::make_shared<T>(x, y_node)},
                                                                  {"Out"});
                    }
                }

                //
                NamedOutputs elementwise_add(const NodeContext& node_context)
                {
                    return elementwise_ops<ngraph::opset6::Add>(node_context);
                }

                NamedOutputs elementwise_sub(const NodeContext& node_context)
                {
                    return elementwise_ops<ngraph::opset6::Subtract>(node_context);
                }

                NamedOutputs elementwise_mul(const NodeContext& node_context)
                {
                    return elementwise_ops<ngraph::opset6::Multiply>(node_context);
                }

                NamedOutputs elementwise_div(const NodeContext& node_context)
                {
                    return elementwise_ops<ngraph::opset6::Divide>(node_context);
                }

                NamedOutputs elementwise_min(const NodeContext& node_context)
                {
                    return elementwise_ops<ngraph::opset6::Minimum>(node_context);
                }

                NamedOutputs elementwise_max(const NodeContext& node_context)
                {
                    return elementwise_ops<ngraph::opset6::Maximum>(node_context);
                }

                NamedOutputs elementwise_pow(const NodeContext& node_context)
                {
                    return elementwise_ops<ngraph::opset6::Power>(node_context);
                }

                NamedOutputs elementwise_equal(const NodeContext& node_context)
                {
                    return elementwise_ops<ngraph::opset6::Equal>(node_context);
                }

                NamedOutputs elementwise_greater_equal(const NodeContext& node_context)
                {
                    return elementwise_ops<ngraph::opset6::GreaterEqual>(node_context);
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph