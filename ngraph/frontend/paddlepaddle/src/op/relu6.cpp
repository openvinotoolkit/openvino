// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "relu6.hpp"
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
                NamedOutputs relu6(const NodeContext& node)
                {
                    auto data = node.get_ng_input("X");
                    auto threshold = node.get_attribute<float>("threshold", 6.0f);
                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::Clamp>(data, 0.0, threshold)}, {"Out"});
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph