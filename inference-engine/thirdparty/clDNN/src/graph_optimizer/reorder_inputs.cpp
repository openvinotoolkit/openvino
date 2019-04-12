/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////


#include "api/CPP/proposal.hpp"
#include "api/CPP/roi_pooling.hpp"
#include "api/CPP/reorg_yolo.hpp"
#include "api/CPP/eltwise.hpp"
#include "upsampling_inst.h"
#include "pass_manager.h"
#include "program_node.h"
#include "layout_optimizer.h"
#include "program_impl.h"
#include "program_helpers.h"

using namespace cldnn;

//ToDo remove friendship relation from program_impl

reorder_inputs::reorder_inputs(layout_optimizer& lo_ref) : base_pass("reorder_inputs"), _lo(lo_ref) {}

void reorder_inputs::run(program_impl& p) {
    run(p, _lo);
}

void reorder_inputs::run(program_impl& p, layout_optimizer& lo)
{
    //first pass to set layout optimization_attributes for topology
    for (auto& node : p.get_processing_order())
    {
        auto& prim = *node;
        if (prim.type() == cldnn::convolution::type_id())
        {
            if (prim.as<convolution>().get_primitive()->split() > 1)
                lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::splitted_convolution, 1);
        }

        //list of layers that do not support yxfb or perform worse than bfyx
        if (prim.type() == cldnn::detection_output::type_id() || prim.type() == cldnn::proposal::type_id() ||
            prim.type() == cldnn::roi_pooling::type_id() || prim.type() == cldnn::deconvolution::type_id() ||
            prim.type() == cldnn::upsampling::type_id() || prim.type() == cldnn::reorg_yolo::type_id())
            lo.set_optimization_attribute(layout_optimizer::optimization_attributes_type::bfyx_only_layer, 1);
    }

    const auto reorder_input = [&p, &lo](typed_program_node<convolution>& conv_node)
    {
        auto conv_prim = conv_node.get_primitive();
        auto& input_node = conv_node.get_dependency(0);
        auto&& weights_layout = conv_node.weights(0).get_output_layout();
        auto&& input_layout = input_node.get_output_layout();

        std::shared_ptr<reorder> new_input = nullptr;

        if (input_node.type() == reorder::type_id()) //convolution's input is a reorder
        {
            auto reorder_prim = input_node.as<reorder>().typed_desc();
            auto& reorder_input = input_node.get_dependency(0);
            auto reorder_layout = input_node.get_output_layout();
            reorder_layout.data_type = *reorder_prim->output_data_type;
            new_input = lo.get_reorder(
                reorder_layout,
                reorder_prim->id,
                layout_optimizer::data_type::input,
                conv_node,
                weights_layout).first;

            auto reorder_removed = false;
            if (new_input && new_input->output_format != format::winograd_2x3_s1_data && new_input->output_format != format::bf8_xy16 && new_input->output_format != format::byxf) //output format is not optimal
            {
                auto reorder_input_layout = reorder_input.get_output_layout();

                auto opt_layout = layout(*new_input->output_data_type, new_input->output_format, reorder_input_layout.size);
                if (reorder_input_layout == opt_layout) //reorder 'breaks' optimal format
                {
                    if (reorder_prim->subtract_per_feature.empty() &&
                        reorder_prim->mean.empty() &&
                        !reorder_prim->output_padding) //just plain reorder
                    {
                        conv_node.replace_dependency(0, reorder_input);
                        if (input_node.get_users().size() == 0 && !input_node.is_output())
                        {
                            reorder_removed = p.extract_and_remove(input_node);
                        }
                        new_input = nullptr;
                    }
                    else //change reorder's output layout
                    {
                        reorder_prim->output_format = opt_layout.format;
                        reorder_prim->output_data_type = opt_layout.data_type;
                        new_input = nullptr;
                    }
                }
                else //current reorder gives bad output, simply change it
                {
                    reorder_prim->output_format = opt_layout.format;
                    reorder_prim->output_data_type = opt_layout.data_type;
                    new_input = nullptr;
                }
            }

            if (!reorder_removed)
                input_node.recalc_output_layout();
            else
                conv_node.recalc_output_layout();
        }
        else
        {
            new_input = lo.get_reorder(
                input_node.get_output_layout(),
                input_node.id(),
                layout_optimizer::data_type::input,
                conv_node,
                weights_layout).first;
        }

        if (new_input && new_input->output_format == format::winograd_2x3_s1_data)
        {
            auto lower_size = (conv_prim->input_offset.negate() + input_layout.size);

            tensor upper_input_padding = tensor{ 0 };
            upper_input_padding.spatial[0] = (2 - (lower_size.spatial[0] % 2)) % 2;          //winograd conv requires input's x to be in form 4 + 2n, with restriction that x >= 3, we can shortage it to x % 2 == 0
            upper_input_padding.spatial[1] = (8 - ((lower_size.spatial[1] - 2) % 8)) % 8;    //for y, y - 2 % 8 == 0 must hold

            p.apply_needed_padding(conv_node, input_node, padding{ conv_prim->input_offset.negate().sizes(), upper_input_padding.sizes() });

            auto winograd_output = std::make_shared<reorder>("_winograd_" + conv_node.id(), conv_node.id(), input_layout.format,
                input_layout.data_type, std::vector<float>{}, cldnn_reorder_mean_mode::mean_subtract, conv_node.output_layout.data_padding);
            conv_node.output_layout.data_padding = padding{};
            program_node& back_node = p.get_or_create(winograd_output);
            p.get_processing_order().insert_next(&conv_node, &back_node);

            auto bias_term = conv_node.bias_term();
            //create additional eltwise node after reorder to compute bias
            if (bias_term)
            {
                auto& bias_node = conv_node.get_dependency(2);
                std::vector<primitive_id> inputs = { back_node.id(), bias_node.id() };
                auto winograd_output_biases = std::make_shared<eltwise>(back_node.id() + "_bias", inputs,
                    cldnn::eltwise_mode::sum, conv_prim->with_activation, conv_prim->activation_negative_slope,
                    back_node.get_output_layout().data_padding);
                back_node.get_output_layout().data_padding = padding{};
                auto& back_bias_node = p.get_or_create(winograd_output_biases);
                p.get_processing_order().insert_next(&back_node, &back_bias_node);
                p.replace_all_usages(back_node, back_bias_node);
                p.add_connection(back_node, back_bias_node);
                p.add_connection(bias_node, back_bias_node);
                conv_node.invalidate_users();
                p.replace_all_usages(conv_node, back_bias_node);
            }

            if (conv_prim->with_activation)
            {
                conv_node.typed_desc()->with_activation = false;
                if (!bias_term)
                    back_node.set_fused_activation(activation_relu_negative_slope, cldnn_activation_additional_params_t{ conv_prim->activation_negative_slope });
            }

            if (!bias_term)
            {
                conv_node.invalidate_users();
                p.replace_all_usages(conv_node, back_node);
            }
            p.add_connection(conv_node, back_node);

            auto& r_node = p.get_or_create(new_input);
            r_node.as<reorder>().set_input_offset(conv_prim->input_offset);

            if (!bias_term)
            {
                p.swap_names(conv_node, back_node);
                if (conv_node.is_output())
                {
                    conv_node.set_output(false);
                    back_node.set_output(true);
                    for (auto& output : p.get_outputs())
                    {
                        if (output == &conv_node)
                        {
                            output = &back_node;
                            break;
                        }
                    }
                }
            }
            else
            {
                conv_node.remove_dependency(2);
                auto& back_bias_node = *(p.nodes_map.find(back_node.id() + "_bias")->second);
                p.swap_names(conv_node, back_bias_node);
                if (conv_node.is_output())
                {
                    conv_node.set_output(false);
                    back_bias_node.set_output(true);
                    for (auto& output : p.get_outputs())
                    {
                        if (output == &conv_node)
                        {
                            output = &back_bias_node;
                            break;
                        }
                    }
                }
            }
        }

        if (new_input && (new_input->output_format == format::bf8_xy16 || new_input->output_format == format::byxf))
        {
            auto conv1x1_output = std::make_shared<reorder>("_conv1x1_reorder_back_" + conv_node.id(), conv_node.id(), input_layout.format, input_layout.data_type);
            auto& back_node = p.get_or_create(conv1x1_output);
            p.get_processing_order().insert_next(&conv_node, &back_node);
            conv_node.invalidate_users();
            p.replace_all_usages(conv_node, back_node);
            p.add_connection(conv_node, back_node);
        }

        if (new_input)
        {
            auto& r_node = p.get_or_create(new_input);
            p.add_intermediate(r_node, conv_node, 0, r_node.get_dependencies().empty());
            conv_node.recalc_output_layout();
        }
    };

    const auto reorder_input_detection_output = [&p, &lo](typed_program_node<detection_output>& detection_output_node)
    {
        auto detection_output_prim = detection_output_node.get_primitive();

        for (size_t i = 0; i < detection_output_node.get_dependencies().size(); i++)
        {
            auto& input = detection_output_node.get_dependency(i);
            std::shared_ptr<reorder> new_input = lo.get_reorder(
                input.get_output_layout(),
                input.id(),
                layout_optimizer::data_type::input,
                detection_output_node,
                layout{ data_types::f32, format::bfyx, tensor{} }).first;

            if (new_input)
            {
                p.add_intermediate(new_input, detection_output_node, i);
            }
        }
    };

    for (auto& prim : p.get_processing_order())
    {
        //there's an assumption that only convolution will take data/input_layout as input
        //exception to that rule would be a convolution which takes a reorder as input - see reoder_input above
        program_helpers::do_for_types<convolution, detection_output>(*prim,
            reorder_input,                  //case for convolution
            reorder_input_detection_output  //case for detection-output
            );
    }
}
