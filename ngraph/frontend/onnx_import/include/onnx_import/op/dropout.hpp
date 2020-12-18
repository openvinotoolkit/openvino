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
#include "onnx_import/default_opset.hpp"
#include "onnx_import/core/node.hpp"
#include "onnx_import/core/null_node.hpp"

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
                    std::cout << "\n ONNX Dropout 12 \n";  
                  
                    auto inputs_size = node.get_ng_inputs().size();
                    auto outputs_size = node.get_outputs_size();
                    auto input_data = node.get_ng_inputs().at(0);

                    if (inputs_size == 1 && outputs_size == 1)
                    {
                        return {input_data};
                    }
                    else
                    {
                        std::shared_ptr<ngraph::Node> shape_of_data;
                        
                        if (input_data.get_partial_shape().is_static())
                        {
                            shape_of_data = default_opset::Constant::create(
                                        ngraph::element::i32, Shape{input_data.get_partial_shape().rank().get_length()}, input_data.get_partial_shape().to_shape());
                        }
                        else 
                        {
                            shape_of_data = std::make_shared<default_opset::ShapeOf>(input_data);
                        }

                        auto mask_value_node = 
                                    default_opset::Constant::create(
                                        input_data.get_element_type(), Shape{}, {1});

                        auto mask_node = 
                                    std::make_shared<default_opset::Broadcast>(mask_value_node, shape_of_data);

                        std::shared_ptr<default_opset::Concat> concat;
                        std::shared_ptr<default_opset::Split> split;

                        if(inputs_size == 1)
                        {
                            concat = std::make_shared<default_opset::Concat>(OutputVector{input_data, mask_node}, 0);
                            split = std::make_shared<default_opset::Split>(concat, default_opset::Constant::create(
                                            ngraph::element::i32, Shape{}, {0}), 2);
                        }
                        else if (inputs_size == 2)
                        {
                            auto input_r = node.get_ng_inputs().at(1);
                            auto broadcast_r = std::make_shared<default_opset::Broadcast>(input_r, shape_of_data);

                            auto convert_r = std::make_shared<default_opset::Convert>(
                                    broadcast_r,
                                input_data.get_element_type());

                            concat = std::make_shared<default_opset::Concat>(OutputVector{input_data, mask_node, convert_r}, 0);
                                
                            split = std::make_shared<default_opset::Split>(concat, default_opset::Constant::create(
                                            ngraph::element::i32, Shape{}, {0}), 3);
                        }
                        else if (inputs_size == 3)
                        {
                            auto input_r = node.get_ng_inputs().at(1);
                            auto input_t = node.get_ng_inputs().at(2);

                            auto broadcast_r = std::make_shared<default_opset::Broadcast>(input_r, shape_of_data);
                            auto broadcast_t = std::make_shared<default_opset::Broadcast>(input_t, shape_of_data);

                            auto convert_r = std::make_shared<default_opset::Convert>(
                                    broadcast_r,
                                input_data.get_element_type());

                            auto convert_t = std::make_shared<default_opset::Convert>(
                                    broadcast_t,
                                input_data.get_element_type());

                            concat = std::make_shared<default_opset::Concat>(OutputVector{input_data, mask_node, convert_r, convert_t}, 0);
                                
                            split = std::make_shared<default_opset::Split>(concat, default_opset::Constant::create(
                                            ngraph::element::i32, Shape{}, {0}), 4);
                        }
                        auto output_mask = std::make_shared<default_opset::Convert>(
                                    split->output(1),
                                ngraph::element::boolean);

                        if (outputs_size > 1) 
                        {
                            return OutputVector{split->output(0), output_mask};
                        }
                        else
                        {
                           return OutputVector{split->output(0)};
                        }
                        

                        // auto output_z = std::make_shared<default_opset::Convert>(
                        //             default_opset::Constant::create(
                        //                 ngraph::element::boolean, input_x.get_shape(), mask),
                        //         ngraph::element::boolean);


                        
                        // auto output_z = std::make_shared<default_opset::Convert>(
                                    
                        //             mask_node,
                        //         ngraph::element::boolean);

                        // auto mask = std::vector<char>(shape_size(input_x.get_shape()), 1);
                        // for (auto m : mask)
                        // {
                        //     std::cout << m << ", \n";
                        // }


                        // Output<ngraph::Node> mask_node = 
                        //             default_opset::Constant::create(
                        //                 ngraph::element::boolean, input_x.get_shape(), mask);


                        // return {input_x, mask_node};

                        // return {split->output(0), split->output(1)};
                        // return {input_x, output_z};


                        
                    //     auto output_z = std::make_shared<default_opset::Convert>(
                    //                 mask_node,
                    //             ngraph::element::boolean);
                    //     // output_z->set_friendly_name("z");
                    //     // mask_node->set_friendly_name("z");


                    //    auto conv_input_x = std::make_shared<default_opset::Convert>(
                    //                 input_x,
                    //             input_x.get_element_type());


                        // auto mask_node = 
                        //             default_opset::Constant::create(
                        //                 ngraph::element::boolean, input_x.get_shape(), mask);
                        
                        // auto node_x = std::make_shared<Node>(input_x);

                        // auto node_mask_node = std::make_shared<Node>(mask_node);

                        // auto node_x = std::make_shared<NullNode>(input_x);

                        // auto node_mask_node = std::make_shared<NullNode>(mask_node);

                        // auto n_output_z = std::make_shared<default_opset::Add>(
                        //     mask_node, default_opset::Constant::create(
                        //                 ngraph::element::i32, input_x.get_shape(), {0})
                        //                 );

                        // auto output_z = std::make_shared<default_opset::Convert>(
                        //     n_output_z, ngraph::element::boolean);

                        // return {conv_input_x, mask_node};


                        // return { Output<ngraph::Node>(input_x),  Output<ngraph::Node>(output_z)};

                        // return { Output<ngraph::Node>(input_x),  Output<ngraph::Node>(mask_node)};

                        // mask_node->set_friendly_name("M");
                       
                        // auto rest_m = std::make_shared<ngraph::opset5::Result>(mask_node);
                        // auto rest_m = std::make_shared<ngraph::opset5::Result>(output_z);

                        // rest_m->set_friendly_name("M");

                        // auto rest_m = std::make_shared<ngraph::opset5::Result>(output_z);

                        // auto rest_x = std::make_shared<ngraph::opset5::Result>(input_x);
                        // rest_x->set_friendly_name("C");

                        // return {input_x, output_z};
                        // return {rest_x, rest_m};
                        // return {input_x, rest_m};


                        // if(inputs_size > 2)
                        // {
                        //     auto input_r = node.get_ng_inputs().at(1);
                        //     return {std::make_shared<ngraph::opset5::Result>(input_x), output_z, input_r};

                        // }
                        // auto rest_z = std::make_shared<ngraph::opset5::Result>(mask_node);
                        // rest_z->set_friendly_name("z");

                        // return {input_x, rest_z};

                        // return {node.get_ng_inputs().at(0)};

                    }

                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
