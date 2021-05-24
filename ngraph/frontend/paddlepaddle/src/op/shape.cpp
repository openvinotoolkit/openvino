// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape.hpp"
#include <ngraph/opsets/opset6.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs shape(const NodeContext& node)
                {
                    auto data = node.get_ng_input("Input");
                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::ShapeOf>(data)}, {"Out"});
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph