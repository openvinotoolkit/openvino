// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "op/leaky_relu.hpp"

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
