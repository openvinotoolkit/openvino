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

#pragma once

#include <memory>

#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"
#include "onnx_import/core/null_node.hpp"
#include "onnx_import/default_opset.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                inline OutputVector dropout(const Node& node)
                {
                    // First value is actual output of Dropout,
                    // the second one is just a placeholder for optional trailing output.
                    return {node.get_ng_inputs().at(0).get_node_shared_ptr(),
                            std::make_shared<NullNode>()};
                }
            } // namespace set_1

            namespace set_12
            {
                inline OutputVector dropout(const Node& node)
                {
                    auto inputs_size = node.get_ng_inputs().size();
                    auto outputs_size = node.get_outputs_size();
                    auto input_data = node.get_ng_inputs().at(0);

                    if (inputs_size == 1 && outputs_size == 1)
                    {
                        return {input_data};
                    }
                    else
                    {
                        // This is work around for hanging optional inputs and optional output
                        // that are not useful for inference
                        // Inputs -> Concat -> Split -> Outputs
                        std::shared_ptr<ngraph::Node> shape_of_data;

                        if (input_data.get_partial_shape().is_static())
                        {
                            shape_of_data = default_opset::Constant::create(
                                ngraph::element::i32,
                                Shape{input_data.get_partial_shape().rank().get_length()},
                                input_data.get_partial_shape().to_shape());
                        }
                        else
                        {
                            shape_of_data = std::make_shared<default_opset::ShapeOf>(input_data);
                        }

                        auto mask_value_node = default_opset::Constant::create(
                            input_data.get_element_type(), Shape{}, {1});

                        auto mask_node = std::make_shared<default_opset::Broadcast>(mask_value_node,
                                                                                    shape_of_data);

                        std::shared_ptr<default_opset::Concat> concat;
                        std::shared_ptr<default_opset::Split> split;

                        if (inputs_size == 1)
                        {
                            concat = std::make_shared<default_opset::Concat>(
                                OutputVector{input_data, mask_node}, 0);
                            split = std::make_shared<default_opset::Split>(
                                concat,
                                default_opset::Constant::create(ngraph::element::i32, Shape{}, {0}),
                                2);
                        }
                        else if (inputs_size == 2)
                        {
                            auto input_r = node.get_ng_inputs().at(1);
                            auto broadcast_r =
                                std::make_shared<default_opset::Broadcast>(input_r, shape_of_data);

                            auto convert_r = std::make_shared<default_opset::Convert>(
                                broadcast_r, input_data.get_element_type());

                            concat = std::make_shared<default_opset::Concat>(
                                OutputVector{input_data, mask_node, convert_r}, 0);

                            split = std::make_shared<default_opset::Split>(
                                concat,
                                default_opset::Constant::create(ngraph::element::i32, Shape{}, {0}),
                                3);
                        }
                        else if (inputs_size == 3)
                        {
                            auto input_r = node.get_ng_inputs().at(1);
                            auto input_t = node.get_ng_inputs().at(2);

                            auto broadcast_r =
                                std::make_shared<default_opset::Broadcast>(input_r, shape_of_data);
                            auto broadcast_t =
                                std::make_shared<default_opset::Broadcast>(input_t, shape_of_data);

                            auto convert_r = std::make_shared<default_opset::Convert>(
                                broadcast_r, input_data.get_element_type());

                            auto convert_t = std::make_shared<default_opset::Convert>(
                                broadcast_t, input_data.get_element_type());

                            concat = std::make_shared<default_opset::Concat>(
                                OutputVector{input_data, mask_node, convert_r, convert_t}, 0);

                            split = std::make_shared<default_opset::Split>(
                                concat,
                                default_opset::Constant::create(ngraph::element::i32, Shape{}, {0}),
                                4);
                        }
                        auto output_mask = std::make_shared<default_opset::Convert>(
                            split->output(1), ngraph::element::boolean);

                        if (outputs_size > 1)
                        {
                            return OutputVector{split->output(0), output_mask};
                        }
                        else
                        {
                            return OutputVector{split->output(0)};
                        }
                    }
                }
            } // namespace set_12

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
