// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"
#include "utils/variadic.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                inline OutputVector min(const Node& node)
                {
                    return variadic::make_ng_variadic_op<default_opset::Minimum>(
                        node, ngraph::op::AutoBroadcastSpec::NONE);
                }

            } // namespace set_1

            namespace set_8
            {
                inline OutputVector min(const Node& node)
                {
                    return variadic::make_ng_variadic_op<default_opset::Minimum>(node);
                }

            } // namespace set_8

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
