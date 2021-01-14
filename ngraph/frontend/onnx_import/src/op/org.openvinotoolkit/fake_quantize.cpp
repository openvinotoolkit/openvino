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

#include "onnx_import/default_opset.hpp"
#include "onnx_import/op/org.openvinotoolkit/fake_quantize.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector fake_quantize(const onnx_import::Node& node)
                {
                    const auto inputs = node.get_ng_inputs();
                    const auto X = inputs.at(0);
                    const auto input_low = inputs.at(1);
                    const auto input_high = inputs.at(2);
                    const auto output_low = inputs.at(3);
                    const auto output_high = inputs.at(4);

                    const auto levels = node.get_attribute_value<std::size_t>("levels");

                    return {std::make_shared<default_opset::FakeQuantize>(
                        X, input_low, input_high, output_low, output_high, levels)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
