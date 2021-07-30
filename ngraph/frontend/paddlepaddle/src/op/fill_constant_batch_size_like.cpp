// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits.h>
#include <ngraph/opsets/opset6.hpp>
#include <node_context.hpp>
#include <paddlepaddle_frontend/utility.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                static std::shared_ptr<Node> get_val(int32_t idx, const Output<Node>& data)
                {
                    auto startsNode = ngraph::opset6::Constant::create(element::i32, {1}, {idx});
                    auto endsNode = ngraph::opset6::Constant::create(element::i32, {1}, {idx + 1});
                    auto stridesNode = ngraph::opset6::Constant::create(element::i32, {1}, {1});
                    return std::make_shared<ngraph::opset6::StridedSlice>(
                        data,
                        startsNode,
                        endsNode,
                        stridesNode,
                        std::vector<int64_t>(1, 0),
                        std::vector<int64_t>(1, 0));
                }

                static std::shared_ptr<Node> set_val(int32_t idx,
                                                     std::shared_ptr<Node> val_node,
                                                     std::shared_ptr<Node> array_node)
                {
                    NodeVector nodes;
                    if (idx > 0)
                    {
                        // [0, idx)
                        auto startsNode = ngraph::opset6::Constant::create(element::i32, {1}, {0});
                        auto endsNode = ngraph::opset6::Constant::create(element::i32, {1}, {idx});
                        auto stridesNode = ngraph::opset6::Constant::create(element::i32, {1}, {1});
                        auto head = std::make_shared<ngraph::opset6::StridedSlice>(
                            array_node,
                            startsNode,
                            endsNode,
                            stridesNode,
                            std::vector<int64_t>(1, 0),
                            std::vector<int64_t>(1, 0));
                        nodes.push_back(head);
                    }
                    nodes.push_back(val_node);
                    // [idx + 1, max)
                    auto startsNode =
                        ngraph::opset6::Constant::create(element::i32, {1}, {idx + 1});
                    auto endsNode = ngraph::opset6::Constant::create(element::i32, {1}, {INT_MAX});
                    auto stridesNode = ngraph::opset6::Constant::create(element::i32, {1}, {1});
                    auto tail =
                        std::make_shared<ngraph::opset6::StridedSlice>(array_node,
                                                                       startsNode,
                                                                       endsNode,
                                                                       stridesNode,
                                                                       std::vector<int64_t>(1, 0),
                                                                       std::vector<int64_t>(1, 0));
                    nodes.push_back(tail);

                    return std::make_shared<ngraph::opset6::Concat>(nodes, 0);
                }

                static Output<Node> get_seed_node(const NodeContext& node)
                {
                    auto dtype = node.get_attribute<element::Type>("dtype");
                    Output<Node> val_node;
                    auto str_value = node.get_attribute<std::string>("str_value");
                    switch (dtype)
                    {
                    case element::i32:
                        val_node =
                            ngraph::opset6::Constant::create(dtype, {1}, {std::stoi(str_value)});
                        break;
                    case element::i64:
                        val_node =
                            ngraph::opset6::Constant::create(dtype, {1}, {std::stoll(str_value)});
                        break;
                    case element::f32:
                        val_node =
                            ngraph::opset6::Constant::create(dtype, {1}, {std::stof(str_value)});
                        break;
                    case element::f64:
                        val_node =
                            ngraph::opset6::Constant::create(dtype, {1}, {std::stod(str_value)});
                        break;
                    default:
                        throw std::runtime_error(
                            "fill_constant_batch_size_like: dtype value is invalid");
                    }

                    return val_node;
                }

                NamedOutputs fill_constant_batch_size_like(const NodeContext& node)
                {
                    auto input_dim_idx = node.get_attribute<int32_t>("input_dim_idx");
                    auto output_dim_idx = node.get_attribute<int32_t>("output_dim_idx");
                    auto shapes = node.get_attribute<std::vector<int32_t>>("shape");
                    auto input = node.get_ng_input("Input");
                    auto input_shape =
                        std::make_shared<ngraph::opset6::ShapeOf>(input, element::i32);
                    // 1, cat the array:
                    //   shape[0, shape[output_dim_idx]) + input_shape[input_dim_idx] +
                    //   shape[shape[output_dim_idx + 1], -1]
                    auto input_val_node = get_val(input_dim_idx, input_shape);
                    auto shapes_node = ngraph::opset6::Constant::create(
                        ngraph::element::i32, {shapes.size()}, shapes);
                    auto shape_node = set_val(output_dim_idx, input_val_node, shapes_node);

                    // 2, use the shape broadcast the node
                    auto val_node = get_seed_node(node);
                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::Broadcast>(val_node, shape_node)},
                        {"Out"});
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph