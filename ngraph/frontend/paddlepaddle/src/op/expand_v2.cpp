// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "expand_v2.hpp"
#include <ngraph/opsets/opset6.hpp>
#include <paddlepaddle_frontend/utility.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs expand_v2(const NodeContext& node)
                {
                    auto x = node.get_ng_input("X");
                    Output<Node> shape_expected_node;
                    if (node.has_ng_input("Shape"))
                    {
                        shape_expected_node = node.get_ng_input("Shape");
                    }
                    else if (node.has_ng_input("expand_shapes_tensor"))
                    {
                        auto inputs = node.get_ng_inputs("expand_shapes_tensor");
                        ngraph::NodeVector node_vec;
                        for (auto& input : inputs)
                        {
                            auto cast =
                                std::make_shared<ngraph::opset6::Convert>(input, element::i32);
                            node_vec.push_back(cast);
                        }
                        shape_expected_node = std::make_shared<ngraph::opset6::Concat>(node_vec, 0);
                    }
                    else
                    {
                        std::vector<int32_t> shape_expected;
                        if (node.has_attribute<std::vector<int32_t>>("shape"))
                        {
                            shape_expected = node.get_attribute<std::vector<int32_t>>("shape");
                        }
                        else
                        {
                            throw std::runtime_error("expand: has no shape attribute");
                        }
                        shape_expected_node = ngraph::opset6::Constant::create(
                            ngraph::element::i32, {shape_expected.size()}, shape_expected);
                    }
                    // if -1 in shape we will copy the orginal value from input
                    auto zero_node =
                        ngraph::opset6::Constant::create(ngraph::element::i32, {1}, {0});
                    auto mask_node =
                        std::make_shared<ngraph::opset6::Greater>(shape_expected_node, zero_node);
                    auto input_shape_node =
                        std::make_shared<ngraph::opset6::ShapeOf>(x, element::i32);
                    auto fixed_shape_node = std::make_shared<ngraph::opset6::Select>(
                        mask_node, shape_expected_node, input_shape_node);

                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::Broadcast>(x, fixed_shape_node)},
                        {"Out"});
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph