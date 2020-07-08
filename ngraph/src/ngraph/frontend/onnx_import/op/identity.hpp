//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <memory>

#include "core/node.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/get_output_element.hpp"

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
                    if (input->get_element_type() == ngraph::element::boolean)
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
