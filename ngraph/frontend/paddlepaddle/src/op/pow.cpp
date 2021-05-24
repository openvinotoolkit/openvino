// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "pow.hpp"
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
                NamedOutputs pow(const NodeContext& node)
                {
                    auto x = node.get_ng_input("X");
                    Output<Node> factor_node;
                    if (node.has_ng_input("FactorTensor"))
                    {
                        factor_node = node.get_ng_input("FactorTensor");
                    }
                    else
                    {
                        auto factor = node.get_attribute<float>("factor");
                        factor_node =
                            ngraph::opset6::Constant::create(ngraph::element::f32, {}, {factor});
                    }

                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::Power>(x, factor_node)}, {"Out"});
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph