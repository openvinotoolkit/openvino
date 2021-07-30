// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

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
                inline OutputVector identity(const Node& node)
                {
                    auto input = node.get_ng_inputs().at(0);
                    if (input.get_element_type() == ngraph::element::boolean)
                    {
                        const auto logic_zero =
                            default_opset::Constant::create(ngraph::element::boolean, {}, {false});
                        return {std::make_shared<default_opset::LogicalOr>(input, logic_zero)};
                    }
                    const auto zero =
                        default_opset::Constant::create(input.get_element_type(), {}, {0});
                    return {std::make_shared<default_opset::Add>(input, zero)};
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
