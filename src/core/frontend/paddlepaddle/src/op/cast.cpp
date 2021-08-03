// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include <node_context.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs cast(const NodeContext& node)
                {
                    auto data = node.get_ng_input("X");
                    auto out_dtype = node.get_attribute<ngraph::element::Type>("out_dtype");

                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::Convert>(data, out_dtype)}, {"Out"});
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph