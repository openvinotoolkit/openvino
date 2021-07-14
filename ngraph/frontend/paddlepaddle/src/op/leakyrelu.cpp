// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "leakyrelu.hpp"
#include <ngraph/opsets/opset6.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs leaky_relu(const NodeContext& node)
                {
                    auto data = node.get_ng_input("X");
                    auto alpha = ngraph::opset6::Constant::create(
                        ngraph::element::f32, {1}, {node.get_attribute<float>("alpha")});
                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::PRelu>(data, alpha)}, {"Out"});
                }
            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph