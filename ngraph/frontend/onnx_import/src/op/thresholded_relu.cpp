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
#include <vector>

#include "onnx_import/default_opset.hpp"
#include "thresholded_relu.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector thresholded_relu(const Node& node)
                {
                    const auto data = node.get_ng_inputs().at(0);
                    const double alpha = node.get_attribute_value<double>("alpha", 1.0);

                    const auto alpha_node =
                        default_opset::Constant::create(data.get_element_type(), Shape{}, {alpha});

                    const auto data_map = std::make_shared<default_opset::Convert>(
                        std::make_shared<default_opset::Greater>(data, alpha_node),
                        data.get_element_type());

                    return {std::make_shared<default_opset::Multiply>(data, data_map)};
                }

            } // namespace set_1default_opset

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
