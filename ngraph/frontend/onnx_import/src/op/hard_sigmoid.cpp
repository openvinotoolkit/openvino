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

#include <memory>

#include "hard_sigmoid.hpp"
#include "onnx_import/default_opset.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector hard_sigmoid(const Node& node)
                {
                    const auto data = node.get_ng_inputs().at(0);

                    const auto alpha = default_opset::Constant::create<double>(
                        data.get_element_type(),
                        Shape{},
                        std::vector<double>{node.get_attribute_value<double>("alpha", 0.2)});

                    const auto beta = default_opset::Constant::create<double>(
                        data.get_element_type(),
                        Shape{},
                        std::vector<double>{node.get_attribute_value<double>("beta", 0.5)});

                    return {std::make_shared<default_opset::HardSigmoid>(data, alpha, beta)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
