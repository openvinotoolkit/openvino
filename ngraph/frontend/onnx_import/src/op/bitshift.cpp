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

#include "op/bitshift.hpp"
#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector bitshift(const Node& node)
                {
                    const Output<ngraph::Node> input_x = node.get_ng_inputs().at(0);
                    const Output<ngraph::Node> input_y = node.get_ng_inputs().at(1);

                    std::string direction = node.get_attribute_value<std::string>("direction", "");

                    CHECK_VALID_NODE(node,
                                     !direction.empty(),
                                     "Required attribute 'direction' is not specified.");

                    CHECK_VALID_NODE(node,
                                     direction == "LEFT" || direction == "RIGHT",
                                     "Only values 'LEFT' and 'RIGHT' are supported for 'direction' "
                                     "attribute. Given: ",
                                     direction);

                    auto shift = std::make_shared<default_opset::Power>(
                        default_opset::Constant::create(input_y.get_element_type(), Shape{1}, {2}),
                        input_y);

                    if (direction == "RIGHT")
                    {
                        return {std::make_shared<default_opset::Divide>(input_x, shift)};
                    }
                    else
                    {
                        return {std::make_shared<default_opset::Multiply>(input_x, shift)};
                    }
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
