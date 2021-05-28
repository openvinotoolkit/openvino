// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape2.hpp"

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs reshape2(const NodeContext& node)
                {
                    auto data = node.get_ng_input("X");
                    if (!node.has_ng_input("Shape") && !node.has_ng_input("ShapeTensor"))
                    {
                        auto shape_attr = node.get_attribute<std::vector<int32_t>>("shape");
                        auto shape_node = ngraph::opset6::Constant::create(
                            ngraph::element::i32, {shape_attr.size()}, shape_attr);
                        return node.default_single_output_mapping(
                            {std::make_shared<ngraph::opset6::Reshape>(data, shape_node, true)},
                            {"Out"});
                    }
                    else
                    {
                        FRONT_END_NOT_IMPLEMENTED("reshape2 with shape as input");
                    }
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph