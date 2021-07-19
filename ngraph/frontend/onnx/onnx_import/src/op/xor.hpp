// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                inline OutputVector logical_xor(const Node& node)
                {
                    return {std::make_shared<default_opset::LogicalXor>(
                        node.get_ng_inputs().at(0),
                        node.get_ng_inputs().at(1),
                        ngraph::op::AutoBroadcastSpec(ngraph::op::AutoBroadcastType::NUMPY))};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
