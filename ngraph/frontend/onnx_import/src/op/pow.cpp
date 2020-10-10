//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include "onnx_import/op/pow.hpp"
#include <memory>
#include "ngraph/node.hpp"
#include "onnx_import/default_opset.hpp"

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
                    auto base_type = inputs[0].get_element_type();
                    std::shared_ptr<ngraph::Node> exponent;
                    if (inputs[1].get_element_type() != base_type)
                    {
                        exponent = std::make_shared<default_opset::Convert>(inputs[1], base_type);
                    }
                    else
                    {
                        exponent = inputs[1].get_node_shared_ptr();
                    }
                    return {std::make_shared<default_opset::Power>(inputs[0], exponent)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
