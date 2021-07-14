// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conv2d_transpose.hpp"
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
                NamedOutputs conv2d_transpose(const NodeContext& node)
                {
                    return conv2d_base<opset6::GroupConvolutionBackpropData,
                                       opset6::ConvolutionBackpropData>(node);
                }

            } // namespace op
        }     // namespace pdpd
    }         // namespace frontend
} // namespace ngraph
