// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scale.hpp"
#include <ngraph/opsets/opset6.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs scale(const NodeContext& node)
                {
                    auto data = node.get_ng_input("X");
                    auto scale = ngraph::opset6::Constant::create(
                        ngraph::element::f32, {1}, {node.get_attribute<float>("scale")});
                    auto bias = ngraph::opset6::Constant::create(
                        ngraph::element::f32, {1}, {node.get_attribute<float>("bias")});
                    auto bias_after_scale = node.get_attribute<bool>("bias_after_scale");
                    auto fp32_data = std::make_shared<ngraph::opset6::Convert>(data, element::f32);
                    if (!bias_after_scale)
                    {
                        auto node_add = std::make_shared<ngraph::opset6::Add>(fp32_data, bias);
                        return node.default_single_output_mapping(
                            {std::make_shared<ngraph::opset6::Multiply>(node_add, scale)}, {"Out"});
                    }
                    else
                    {
                        auto node_multiply =
                            std::make_shared<ngraph::opset6::Multiply>(fp32_data, scale);
                        return node.default_single_output_mapping(
                            {std::make_shared<ngraph::opset6::Add>(node_multiply, bias)}, {"Out"});
                    }
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph