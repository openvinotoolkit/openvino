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
#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "ngraph/validation_util.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                inline OutputVector gather(const Node& node)
                {
                    OutputVector ng_inputs{node.get_ng_inputs()};
                    auto data = ng_inputs.at(0);
                    auto indices = ng_inputs.at(1);
                    auto axis = node.get_attribute_value<int64_t>("axis", 0);
                    const auto valid_axis = ngraph::normalize_axis(
                        node.get_description(), axis, data.get_partial_shape().rank());

                    return {std::make_shared<default_opset::Gather>(
                        data,
                        indices,
                        default_opset::Constant::create(element::i64, Shape{}, {valid_axis}))};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
