// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "expand_v2.hpp"
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
                NamedOutputs expand_v2(const NodeContext& node)
                {
                    auto x = node.get_ng_input("X");
                    Output<Node> shapeExpectedNode;
                    if (node.has_ng_input("Shape"))
                    {
                        shapeExpectedNode = node.get_ng_input("Shape");
                    }
                    else
                    {
                        std::vector<int32_t> shapeExpected;
                        if (node.has_attribute<std::vector<int32_t>>("shape"))
                        {
                            shapeExpected = node.get_attribute<std::vector<int32_t>>("shape");
                        }
                        else
                        {
                            throw std::runtime_error("expand: has no shape attribute");
                        }

                        shapeExpectedNode = ngraph::opset6::Constant::create(
                            ngraph::element::i32, {shapeExpected.size()}, shapeExpected);
                    }
                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::Broadcast>(x, shapeExpectedNode)},
                        {"Out"});
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph