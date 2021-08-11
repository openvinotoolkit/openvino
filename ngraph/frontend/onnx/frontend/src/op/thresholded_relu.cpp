// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "default_opset.hpp"
#include "op/thresholded_relu.hpp"

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

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
