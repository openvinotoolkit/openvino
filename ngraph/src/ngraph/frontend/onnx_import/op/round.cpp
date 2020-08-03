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

// Disabled in CMakeList
// Update to higher opset required

#include <memory>

#include "default_opset.hpp"
#include "round.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                // WARNING!
                // Current version is:
                // data_floor = floor(data)
                // diff = data - data_floor
                // if(diff < 0.5f)
                //   return data_floor
                // else
                //   return data_floor + 1.0f
                //
                // The correct version should contain condition:
                // if (diff < 0.5f || (diff == 0.5f && static_cast<int>(data_floor) % 2 == 0))
                OutputVector round(const Node& node)
                {
                    const Output<ngraph::Node> data{node.get_ng_inputs().at(0)};

                    Output<ngraph::Node> one_const =
                        default_opset::Constant::create(element::f32, {}, {1.0f});
                    Output<ngraph::Node> half_const =
                        default_opset::Constant::create(element::f32, {}, {0.5f});
                    if (data.get_element_type() != element::f32)
                    {
                        one_const = std::make_shared<default_opset::Convert>(
                            one_const, data.get_element_type());
                        half_const = std::make_shared<default_opset::Convert>(
                            half_const, data.get_element_type());
                    }

                    const auto data_floor = std::make_shared<default_opset::Floor>(data);
                    const auto data_floor_plus_one =
                        std::make_shared<default_opset::Add>(data_floor, one_const);

                    const auto diff = std::make_shared<default_opset::Add>(
                        data, std::make_shared<default_opset::Negative>(data_floor));
                    const auto less_than_half =
                        std::make_shared<default_opset::Less>(diff, half_const);

                    return {std::make_shared<default_opset::Select>(
                        less_than_half, data_floor, data_floor_plus_one)};
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
