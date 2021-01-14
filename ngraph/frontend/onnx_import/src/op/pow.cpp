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

#include "ngraph/node.hpp"
#include "onnx_import/default_opset.hpp"
#include "onnx_import/op/pow.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector pow(const Node& node)
                {
                    auto inputs = node.get_ng_inputs();
                    NGRAPH_CHECK(inputs.size() == 2,
                                 "Power operation requires 2 inputs. Got: ",
                                 inputs.size());

                    auto base = inputs[0];
                    auto exponent = inputs[1];
                    auto base_type = inputs[0].get_element_type();
                    auto exponent_type = inputs[1].get_element_type();
                    if (exponent_type != base_type)
                    {
                        if (exponent_type.is_integral() ||
                            (base_type.is_real() &&
                             base_type.bitwidth() >= exponent_type.bitwidth()))
                        {
                            exponent =
                                std::make_shared<default_opset::Convert>(exponent, base_type);
                        }
                        else
                        {
                            base = std::make_shared<default_opset::Convert>(base, exponent_type);
                            auto power = std::make_shared<default_opset::Power>(base, exponent);
                            return {std::make_shared<default_opset::Convert>(power, base_type)};
                        }
                    }
                    return {std::make_shared<default_opset::Power>(base, exponent)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
