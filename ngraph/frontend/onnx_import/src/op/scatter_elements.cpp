//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include <memory>

#include "ngraph/opsets/opset3.hpp"
#include "onnx_import/default_opset.hpp"
#include "onnx_import/op/scatter_elements.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector scatter_elements(const Node& node)
                {
                    const auto data = node.get_ng_inputs().at(0);
                    const auto indices = node.get_ng_inputs().at(1);
                    const auto updates = node.get_ng_inputs().at(2);

                    const auto axis = node.get_attribute_value<std::int64_t>("axis", 0);
                    const auto axis_node =
                        default_opset::Constant::create(element::i64, Shape{}, {axis});

                    return {std::make_shared<ngraph::opset3::ScatterElementsUpdate>(
                        data, indices, updates, axis_node)};
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
