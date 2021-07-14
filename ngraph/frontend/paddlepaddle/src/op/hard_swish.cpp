// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hard_swish.hpp"
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
                NamedOutputs hard_swish(const NodeContext& node)
                {
                    auto data = node.get_ng_input("X");
                    if (node.has_attribute<float>("threshold"))
                    {
                        auto threshold = node.get_attribute<float>("threshold");
                        PDPD_ASSERT(std::abs(threshold - 6.0) < 0.001,
                                    "hard_swish: Only threshold = 6.0 is currently supported");
                    }
                    if (node.has_attribute<float>("scale"))
                    {
                        auto scale = node.get_attribute<float>("scale");
                        PDPD_ASSERT(std::abs(scale - 6.0) < 0.001,
                                    "hard_swish: Only scale = 6.0 is currently supported");
                    }
                    if (node.has_attribute<float>("offset"))
                    {
                        auto offset = node.get_attribute<float>("offset");
                        PDPD_ASSERT(std::abs(offset - 3.0) < 0.001,
                                    "hard_swish: Only offset = 3.0 is currently supported");
                    }
                    return node.default_single_output_mapping(
                        {std::make_shared<ngraph::opset6::HSwish>(data)}, {"Out"});
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph