// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <vector>

#include "default_opset.hpp"
#include "ngraph/builder/split.hpp"
#include "op/split.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector split(const Node& node)
                {
                    const auto input = node.get_ng_inputs().at(0);
                    const auto axis = node.get_attribute_value<int64_t>("axis", 0);

                    if (node.has_attribute("split"))
                    {
                        const auto splits =
                            node.get_attribute_value<std::vector<std::size_t>>("split");
                        return ngraph::builder::opset1::split(input, splits, axis);
                    }
                    else
                    {
                        const auto outputs_number = node.get_output_names().size();
                        return ngraph::builder::opset1::split(input, outputs_number, axis);
                    }
                }

            } // namespace set_1

            namespace set_13
            {
                OutputVector split(const Node& node)
                {
                    const auto inputs = node.get_ng_inputs();
                    const auto axis = node.get_attribute_value<int64_t>("axis", 0);

                    if (inputs.size() < 2)
                    {
                        const auto outputs_number = node.get_output_names().size();
                        return ngraph::builder::opset1::split(inputs.at(0), outputs_number, axis);
                    }
                    else
                    {
                        const auto axis_node =
                            default_opset::Constant::create(element::Type_t::i64, Shape{}, {axis});
                        return {std::make_shared<default_opset::VariadicSplit>(
                                    inputs.at(0), axis_node, inputs.at(1))
                                    ->outputs()};
                    }
                }

            } // namespace set_13
        }     // namespace op

    } // namespace onnx_import

} // namespace ngraph
