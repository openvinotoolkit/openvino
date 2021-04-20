// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <memory>

#include "core/null_node.hpp"
#include "default_opset.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "op/clip.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector clip(const Node& node)
                {
                    const auto data = node.get_ng_inputs().at(0);

                    const double max_value =
                        node.get_attribute_value<double>("max", std::numeric_limits<double>::max());

                    const double min_value = node.get_attribute_value<double>(
                        "min", std::numeric_limits<double>::lowest());

                    return {std::make_shared<default_opset::Clamp>(data, min_value, max_value)};
                }

            } // namespace set_1

            namespace set_11
            {
                OutputVector clip(const Node& node)
                {
                    const OutputVector inputs{node.get_ng_inputs()};
                    const Output<ngraph::Node> data = inputs.at(0);
                    const element::Type data_type = data.get_element_type();
                    Output<ngraph::Node> min;
                    Output<ngraph::Node> max;

                    // If second input is provided, assign to min input, otherwise set lowest
                    // numeric limit of double as min input.
                    if (inputs.size() > 1 && !ngraph::op::is_null(inputs.at(1)))
                    {
                        min = inputs.at(1);
                    }
                    else
                    {
                        min = builder::make_constant_from_double(
                            data_type, Shape{}, std::numeric_limits<double>::lowest());
                    }

                    // If third input is provided, assign to max input, otherwise set maximum
                    // numeric limit of double as max input.
                    if (inputs.size() == 3 && !ngraph::op::is_null(inputs.at(2)))
                    {
                        max = inputs.at(2);
                    }
                    else
                    {
                        max = builder::make_constant_from_double(
                            data_type, Shape{}, std::numeric_limits<double>::max());
                    }

                    const auto max_of_min_and_data =
                        std::make_shared<default_opset::Maximum>(min, data);

                    return {std::make_shared<default_opset::Minimum>(max, max_of_min_and_data)};
                }

            } // namespace set_11

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
