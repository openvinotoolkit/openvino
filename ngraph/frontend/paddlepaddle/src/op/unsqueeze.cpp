// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unsqueeze.hpp"
#include <ngraph/opsets/opset6.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs unsqueeze(const NodeContext& node)
                {
                    // TODO to support data type other than int32_t #55168
                    auto data = node.get_ng_input("X");
                    auto axes = node.get_attribute<std::vector<int32_t>>("axes");
                    auto axesNode =
                        ngraph::opset6::Constant::create(ngraph::element::i32, {axes.size()}, axes);
                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::Unsqueeze>(data, axesNode)}, {"Out"});
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph
