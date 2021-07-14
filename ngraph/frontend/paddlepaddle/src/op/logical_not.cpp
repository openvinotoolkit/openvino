// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "logical_not.hpp"
#include <ngraph/opsets/opset6.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs logical_not(const NodeContext& node)
                {
                    auto data = node.get_ng_input("X");
                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::LogicalNot>(data)}, {"Out"});
                }
            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph
