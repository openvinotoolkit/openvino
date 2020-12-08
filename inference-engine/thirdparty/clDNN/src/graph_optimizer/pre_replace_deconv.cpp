/*
// Copyright (c) 2018-2020 Intel Corporation
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

#include "program_helpers.h"
#include "pass_manager.h"

#include "convolution_inst.h"
#include "deconvolution_inst.h"
#include "depth_to_space_inst.h"
#include "kernel_selector_utils.h"
#include <vector>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include "error_handler.h"

void pre_replace_deconv::run(program_impl& p) {
    bool update_processing_order = false;
    auto itr = p.nodes_map.begin();
    while (itr != p.nodes_map.end()) {
        auto node_itr = itr++;
        auto& node = (*node_itr).second;
        // find deconvolution primitives with stride 1 and change them to convolution with trasposed weights
        if (node->is_type<deconvolution>()) {
            if (!p.get_options().get<build_option_type::optimize_data>()->enabled())
                continue;

            auto& deconv_node = node->as<deconvolution>();
            auto& weights_node = deconv_node.weights();
            auto deconv_prim = node->as<deconvolution>().typed_desc();
            tensor filter_size = weights_node.get_output_layout().size;
            auto weights = deconv_prim->weights;

            std::vector<primitive_id> weights_vec;
            for (auto& weights_id : weights)
                weights_vec.push_back(weights_id);

            for (auto& weights_id : weights_vec) {
                auto weights_iter = p.nodes_map.find(weights_id);
                if (weights_iter == p.nodes_map.end())
                    continue;
            }

            // limit optimization to stride = 1
            bool unit_stride = std::all_of(deconv_prim->stride.spatial.begin(),
                                           deconv_prim->stride.spatial.end(),
                                           [](tensor::value_type v) { return v == 1; });
            if (unit_stride) {
                primitive_id deconv_id = node->id();
                auto& input_node = node->get_dependency(0);
                auto groups = deconv_node.get_groups();

                bool perform_opt = false;
                // fp16 and fp32 bfyx implementation supports transposed convolution
                perform_opt |= cldnn::format::dimension(input_node.get_output_layout().format) == 4 &&
                               (input_node.get_output_layout().data_type == data_types::f32 || input_node.get_output_layout().data_type == data_types::f16) &&
                               !((_lo.get_optimization_attributes().b_fs_yx_fsv16_network || input_node.get_output_layout().format == format::b_fs_yx_fsv16) &&
                                _lo.is_format_optimized(node->as<deconvolution>(), format::b_fs_yx_fsv16));
                // int8/uint8 input
                perform_opt |= (input_node.get_output_layout().data_type == data_types::i8 || input_node.get_output_layout().data_type == data_types::u8);

                if (!perform_opt)
                    continue;

                primitive_id input_id = deconv_prim->input[0];

                // setting convolution parameters based on deconvolution params
                auto stride = deconv_prim->stride;
                auto biases = deconv_prim->bias;
                std::vector<primitive_id> bias_vec;
                for (auto& bias_id : biases)
                    bias_vec.push_back(bias_id);
                auto input_offset = deconv_prim->input_offset;
                auto output_padding = deconv_prim->output_padding;

                // remove deconvolution node and its connections to weights and biases, rename it and move to the optimized
                // list
                p.remove_connection(node->get_dependency(0), *node);
                for (auto& weights_id : weights_vec) {
                    auto weights_iter = p.nodes_map.find(weights_id);
                    if (weights_iter == p.nodes_map.end())
                        continue;

                    auto weights_node_ptr = weights_iter->second;
                    p.remove_connection(*weights_node_ptr, *node);
                }

                input_offset.spatial[0] = std::abs(input_offset.spatial[0]) - (filter_size.spatial[0] - 1);
                input_offset.spatial[1] = std::abs(input_offset.spatial[1]) - (filter_size.spatial[1] - 1);
                input_offset.spatial[2] = std::abs(input_offset.spatial[2]) - (filter_size.spatial[2] - 1);

                if (!bias_vec.empty()) {
                    for (auto& bias_id : bias_vec) {
                        auto bias_iter = p.nodes_map.find(bias_id);
                        if (bias_iter == p.nodes_map.end())
                            continue;

                        auto bias_id_node_ptr = bias_iter->second;
                        p.remove_connection(*bias_id_node_ptr, *node);
                    }
                }
                auto rename_id = deconv_id + "_tmp";
                auto was_output = node->is_output();
                if (was_output) {
                    node->set_output(false);
                    auto& outputs = p.get_outputs();
                    outputs.erase(std::remove(outputs.begin(), outputs.end(), node.get()), outputs.end());
                }
                p.rename(*node, rename_id);

                // create convolution primitive
                if (!biases.empty()) {
                    auto conv_prim = std::make_shared<convolution>(deconv_id,
                                                                   input_id,
                                                                   weights_vec,
                                                                   bias_vec,
                                                                   groups,
                                                                   stride,
                                                                   input_offset,
                                                                   tensor{ 1, 1, 1, 1 },
                                                                   output_padding);
                    p.get_or_create(conv_prim);
                } else {
                    auto conv_prim = std::make_shared<convolution>(deconv_id,
                                                                   input_id,
                                                                   weights_vec,
                                                                   groups,
                                                                   stride,
                                                                   input_offset,
                                                                   tensor{ 1, 1, 1, 1 },
                                                                   output_padding);
                    p.get_or_create(conv_prim);
                }

                auto conv_node_itr = p.nodes_map.find(deconv_id);
                if (conv_node_itr == p.nodes_map.end())
                    continue;

                auto conv_node_ptr = conv_node_itr->second;
                auto conv_node = &conv_node_ptr->as<convolution>();
                conv_node->set_transposed(true);

                // add connections input->convolution, weights->convolution and bias->convolution
                p.add_connection(input_node, *conv_node_ptr);

                for (auto& weights_id : weights_vec) {
                    auto weights_node_itr = p.nodes_map.find(weights_id);
                    if (weights_node_itr == p.nodes_map.end())
                        continue;

                    auto weights_node_ptr = weights_node_itr->second;
                    p.add_connection(*weights_node_ptr, *conv_node_ptr);
                }

                if (!bias_vec.empty()) {
                    for (auto& bias_id : bias_vec) {
                        auto bias_id_node_itr = p.nodes_map.find(bias_id);
                        if (bias_id_node_itr == p.nodes_map.end())
                            continue;

                        auto bias_id_node_ptr = bias_id_node_itr->second;
                        p.add_connection(*bias_id_node_ptr, *conv_node_ptr);
                    }
                }

                auto deconv_node_itr = p.nodes_map.find(rename_id);
                if (deconv_node_itr != p.nodes_map.end()) {
                    auto deconv_node_ptr = deconv_node_itr->second;
                    p.replace_all_usages(*deconv_node_ptr, *conv_node_ptr);
                    p.optimized_out.push_back(rename_id);
                    p.nodes_map.erase(rename_id);
                }

                if (was_output) {
                    conv_node->set_output(true);
                    p.get_outputs().push_back(conv_node);
                }

                p.mark_if_data_flow(*conv_node);
                conv_node->recalc_output_layout(true);

                update_processing_order = true;
            // current optimization only available for specific deconvolution parameters
            } else if (node->is_output() == false &&
               node->get_output_layout().size.feature[0] == 1 &&
               deconv_prim->stride.spatial[0] == 2 && deconv_prim->stride.spatial[1] == 2 &&
               filter_size.spatial[0] == 9 && filter_size.spatial[1] == 9 &&
               deconv_prim->input_offset.spatial[0] == -4 && deconv_prim->input_offset.spatial[1] == -4 &&
               weights_vec.size() == 1 && deconv_prim->bias.size() == 1 &&
               node->get_dependency(0).get_output_layout().format == format::bfyx) {
                primitive_id deconv_id = node->id();
                auto& input_node = node->get_dependency(0);
                primitive_id input_id = deconv_prim->input[0];

                auto scale_factor = deconv_prim->stride.spatial[0];

                auto cur_weights_node_ptr = p.nodes_map.find(weights_vec[0])->second;
                auto weights_layout = cur_weights_node_ptr->get_output_layout();
                auto weights_data_type = weights_layout.data_type;

                auto biases = deconv_prim->bias[0];
                auto bias_id_node_ptr = p.nodes_map.find(biases)->second;
                auto bias_data_type = bias_id_node_ptr->get_output_layout().data_type;

                // enable only for fp32 and fp16
                if (weights_data_type != data_types::f16 &&
                    weights_data_type != data_types::f32 &&
                    bias_data_type != data_types::f16 &&
                    bias_data_type != data_types::f32)
                    continue;

                // setting convolution parameters based on deconvolution params
                tensor stride = { 1, 1, 1, 1 };
                tensor input_offset = { 0, 0, -scale_factor, -scale_factor };
                auto output_padding = deconv_prim->output_padding;

                // remove deconvolution node and its connections to weights and biases,
                // rename it and move to the optimized list
                p.remove_connection(node->get_dependency(0), *node);

                auto weights_node_ptr = p.nodes_map.find(weights_vec[0])->second;
                p.remove_connection(*weights_node_ptr, *node);
                p.remove_connection(*bias_id_node_ptr, *node);

                auto rename_id = deconv_id + "_tmp";
                p.rename(*node, rename_id);

                // reshape weights
                int pixel_shuffle_size = scale_factor * scale_factor;
                int kernel_size = 5;
                tensor target_weights_size = { pixel_shuffle_size, filter_size.feature[0], kernel_size, kernel_size };
                auto target_weights_layout = layout{ weights_layout.data_type, weights_layout.format, target_weights_size };

                {
                     memory_impl::ptr data_to_allocate = p.get_engine().allocate_memory(target_weights_layout, 0);

                     std::vector<float> weights_vec_float;

                     if (weights_data_type == data_types::f16) {
                         mem_lock<half_t> src{ cur_weights_node_ptr->as<data>().get_attached_memory() };
                         for (uint32_t i = 0; i < weights_layout.size.count(); i++)
                             weights_vec_float.push_back(static_cast<float>(src.data()[i]));
                     } else {
                         mem_lock<float> src{ cur_weights_node_ptr->as<data>().get_attached_memory() };
                         for (uint32_t i = 0; i < weights_layout.size.count(); i++)
                             weights_vec_float.push_back(src.data()[i]);
                     }

                     std::vector<std::vector<std::vector<float> > > subpixel_weights(pixel_shuffle_size);

                     program_helpers::reshape_deconvolution_weights(weights_vec_float,
                         static_cast<int>(filter_size.feature[0]),
                         static_cast<int>(filter_size.spatial[0]),
                         static_cast<int>(filter_size.spatial[1]),
                         scale_factor,
                         subpixel_weights);

                     if (weights_data_type == data_types::f16) {
                         mem_lock<half_t> dst{ data_to_allocate };
                         program_helpers::set_weights_values<half_t>(dst.data(), subpixel_weights);
                     } else if (weights_data_type == data_types::f32) {
                         mem_lock<float> dst{ data_to_allocate };
                         program_helpers::set_weights_values<float>(dst.data(), subpixel_weights);
                     } else {
                         throw std::logic_error("Not supported data type.");
                     }

                     memory api_memory = memory(data_to_allocate.detach());
                     auto data_node_weights_replace = std::make_shared<data>(weights_vec[0] + "_conv_rpl", api_memory);
                     p.get_or_create(data_node_weights_replace);
                     auto data_node_weights_replace_node_ptr = p.nodes_map.find(weights_vec[0] + "_conv_rpl")->second;
                     auto& data_node = data_node_weights_replace_node_ptr->as<data>();
                     data_node.set_output_layout(target_weights_layout, false);
                }
                float bias = 0;

                if (bias_data_type == data_types::f16) {
                    mem_lock<half_t> src{ bias_id_node_ptr->as<data>().get_attached_memory() };
                    bias = static_cast<float>(src.data()[0]);
                } else {
                    mem_lock<float> src{ bias_id_node_ptr->as<data>().get_attached_memory() };
                    bias = src.data()[0];
                }

                auto deconv_id_conv = deconv_id + "_conv";

                // create convolution primitive
                auto conv_prim = std::make_shared<convolution>(deconv_id_conv,
                    input_id,
                    std::vector<primitive_id>{ weights_vec[0] + "_conv_rpl" },
                    stride,
                    input_offset,
                    tensor{ 1, 1, 1, 1 },
                    output_padding);
                p.get_or_create(conv_prim);

                auto conv_node_itr = p.nodes_map.find(deconv_id_conv);
                if (conv_node_itr == p.nodes_map.end()) continue;

                auto conv_node_ptr = conv_node_itr->second;
                auto conv_node = &conv_node_ptr->as<convolution>();

                // add connections input->convolution, weights->convolution and bias->convolution
                p.add_connection(input_node, *conv_node_ptr);

                {
                    auto weights_node_conv_rpl_ptr = p.nodes_map.find(weights_vec[0] + "_conv_rpl")->second;
                    p.add_connection(*weights_node_conv_rpl_ptr, *conv_node_ptr);
                    p.inputs.push_back(weights_node_conv_rpl_ptr.get());
                }

                auto pixel_shuffle_prim = std::make_shared<depth_to_space>(deconv_id, deconv_id_conv, 2, depth_to_space_mode::blocks_first);

                p.get_or_create(pixel_shuffle_prim);
                auto pixel_shuffle_node_ptr = p.nodes_map.find(deconv_id)->second;
                pixel_shuffle_node_ptr->add_fused_activation(activation_func::linear, { 1, bias });

                // add connections input->convolution, weights->convolution
                p.add_connection(*conv_node_ptr, *pixel_shuffle_node_ptr);

                auto deconv_node_ptr = p.nodes_map.find(rename_id);
                if (deconv_node_ptr != p.nodes_map.end()) {
                    p.replace_all_usages(*deconv_node_ptr->second, *pixel_shuffle_node_ptr);
                    p.optimized_out.push_back(rename_id);
                    p.nodes_map.erase(rename_id);
                }
                p.mark_if_data_flow(*conv_node);
                conv_node->recalc_output_layout(true);

                update_processing_order = true;
            }
        }
    }

    if (update_processing_order) {
        p.get_processing_order().calc_processing_order(p);
    }
}
