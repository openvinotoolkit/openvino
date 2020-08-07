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

#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"
#include "onnx_import/default_opset.hpp"
#include "onnx_import/utils/variadic.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                inline OutputVector max(const Node& node)
                {
                    return variadic::make_ng_variadic_op<default_opset::Maximum>(
                        node, ngraph::op::AutoBroadcastSpec::NONE);
                }

            } // namespace set_1

            namespace set_8
            {
                inline OutputVector max(const Node& node)
                {
                    return variadic::make_ng_variadic_op<default_opset::Maximum>(node);
                }

            } // namespace set_8

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
