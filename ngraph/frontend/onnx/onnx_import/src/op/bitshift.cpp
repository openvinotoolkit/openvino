// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
