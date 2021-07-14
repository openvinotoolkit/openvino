// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape2.hpp"
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
                NamedOutputs reshape2(const NodeContext& node)
                {
                    auto data = node.get_ng_input("X");
                    if (!node.has_ng_input("Shape") && !node.has_ng_input("ShapeTensor"))
                    {
                        auto shape_attr = node.get_attribute<std::vector<int32_t>>("shape");
                        auto shape_node = ngraph::opset6::Constant::create(
                            ngraph::element::i32, {shape_attr.size()}, shape_attr);
                        return node.default_single_output_mapping(
                            {std::make_shared<ngraph::opset6::Reshape>(data, shape_node, true)},
                            {"Out"});
                    }
                    else
                    {
                        std::string name = "Shape";
                        if (node.has_ng_input("ShapeTensor"))
                        {
                            name = "ShapeTensor";
                        }

                        auto nodes = node.get_ng_inputs(name);
                        ngraph::NodeVector node_vec;
                        for (auto& input_node : nodes)
                        {
                            auto cast =
                                std::make_shared<ngraph::opset6::Convert>(input_node, element::i64);
                            node_vec.push_back(cast);
                        }

                        auto shape_node = std::make_shared<ngraph::opset6::Concat>(node_vec, 0);
                        return node.default_single_output_mapping(
                            {std::make_shared<ngraph::opset6::Reshape>(data, shape_node, true)},
                            {"Out"});
                    }
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph