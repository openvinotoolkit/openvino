// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "greater_equal.hpp"
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
                NamedOutputs greater_equal(const NodeContext& node)
                {
                    auto x = node.get_ng_input("X");
                    auto y = node.get_ng_input("Y");
                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::GreaterEqual>(x, y)}, {"Out"});
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph