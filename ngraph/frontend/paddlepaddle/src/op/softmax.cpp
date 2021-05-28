// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax.hpp"
#include <ngraph/opsets/opset6.hpp>

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs softmax(const NodeContext& node)
                {
                    auto data = node.get_ng_input("X");
                    auto axis = node.get_attribute<int32_t>("axis");
                    if (axis < 0)
                    {
                        PDPD_OP_VALIDATION_CHECK(node,
                                                 data.get_partial_shape().rank().is_static(),
                                                 "Softmax rank must be static");
                        auto data_rank = data.get_partial_shape().rank().get_length();
                        axis = data_rank + axis;
                    }
                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::Softmax>(data, axis)}, {"Out"});
                }
            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph