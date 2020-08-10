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

#include "leaky_relu.hpp"
#include "onnx_import/default_opset.hpp"
#include "onnx_import/exceptions.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector leaky_relu(const Node& node)
                {
                    auto data = node.get_ng_inputs().at(0);
                    double alpha = node.get_attribute_value<double>("alpha", 0.01);

                    CHECK_VALID_NODE(
                        node, alpha >= 0 && alpha <= 1, " alpha value should be in range (0,1)");

                    std::shared_ptr<ngraph::Node> alpha_node =
                        default_opset::Constant::create(data.get_element_type(), Shape{}, {alpha});
                    return {std::make_shared<default_opset::PRelu>(data, alpha_node)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
