// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hard_sigmoid.hpp"
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
                NamedOutputs hard_sigmoid(const NodeContext& node)
                {
                    auto data = node.get_ng_input("X");
                    auto slope = 0.2f;
                    if (node.has_attribute<float>("slope"))
                    {
                        slope = node.get_attribute<float>("slope");
                    }
                    auto offset = 0.5f;
                    if (node.has_attribute<float>("offset"))
                    {
                        offset = node.get_attribute<float>("offset");
                    }
                    auto alpha =
                        ngraph::opset6::Constant::create(ngraph::element::f32, {}, {slope});
                    auto beta =
                        ngraph::opset6::Constant::create(ngraph::element::f32, {}, {offset});
                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::HardSigmoid>(data, alpha, beta)},
                        {"Out"});
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph