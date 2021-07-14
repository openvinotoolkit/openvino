// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hard_sigmoid.hpp"
#include <ngraph/builder/make_constant.hpp>
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
                    auto dtype = data.get_element_type();
                    float slope = node.get_attribute<float>("slope", 0.2f);
                    float offset = node.get_attribute<float>("offset", 0.5f);
                    auto alpha = builder::make_constant(dtype, {}, slope);
                    auto beta = builder::make_constant(dtype, {}, offset);
                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::HardSigmoid>(data, alpha, beta)},
                        {"Out"});
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph