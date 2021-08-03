// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include "conv2d_utils.hpp"

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            namespace op
            {
                NamedOutputs conv2d(const NodeContext& node)
                {
                    return conv2d_base<opset6::GroupConvolution, opset6::Convolution>(node);
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph
