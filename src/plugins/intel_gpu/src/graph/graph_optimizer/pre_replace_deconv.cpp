// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "program_helpers.h"
#include "pass_manager.h"

#include "convolution_inst.h"
#include "deconvolution_inst.h"
#include "depth_to_space_inst.h"
#include <vector>
#include <list>
#include <memory>
#include <string>
#include <utility>

using namespace cldnn;

void pre_replace_deconv::run(program& p) {
    bool update_processing_order = false;

    auto& stream = p.get_stream();

    auto& lo = p.get_layout_optimizer();

    auto itr = p.nodes_map.begin();
    while (itr != p.nodes_map.end()) {
        auto node_itr = itr++;
        auto& node = (*node_itr).second;
        // find deconvolution primitives with stride 1 and change them to convolution with transposed weights
        if (node->is_type<deconvolution>()) {
            if (node->is_dynamic())
                continue;

            auto& deconv_node = node->as<deconvolution>();
            auto& weights_node = deconv_node.weights();
            auto deconv_prim = deconv_node.typed_desc();
            auto filter_layout = weights_node.get_output_layout().convert_to_weights_layout(deconv_prim->grouped_weights_shape);
            auto weights_nodes_id = deconv_prim->weights;
            auto biases_nodes_id = deconv_prim->bias;
            auto& input_node = deconv_node.get_dependency(0);
            auto input_layout = deconv_node.get_input_layout(0);
            const primitive_id deconv_node_id = deconv_node.id();
            const primitive_id& input_node_id = input_node.id();

            // limit optimization to stride = 1
            // iterators shouldn't be used here because of incorrect iterator functionality in mutable_array_ref<>
            bool unit_stride = all_ones(deconv_prim->stride);
            if (unit_stride) {
                auto groups = deconv_node.get_groups();

                bool perform_opt = false;
                // fp16 and fp32 bfyx implementation supports transposed convolution
                perform_opt |= cldnn::format::dimension(input_layout.format) == 4 &&
                               (input_layout.data_type == data_types::f32 || input_layout.data_type == data_types::f16) &&
                               !((lo.get_optimization_attributes().b_fs_yx_fsv16_network || input_layout.format == format::b_fs_yx_fsv16) &&
                                lo.is_format_optimized(deconv_node, format::b_fs_yx_fsv16));
                // int8/uint8 input
                perform_opt |= (input_layout.data_type == data_types::i8 || input_layout.data_type == data_types::u8);

                if (!perform_opt)
                    continue;

                // setting convolution parameters based on deconvolution params
                auto output_layout = deconv_node.get_output_layout();
                auto output_pshape = output_layout.get_partial_shape();
                auto input_pshape = input_layout.get_partial_shape();
                auto spatial_rank = output_layout.get_spatial_rank();
                auto stride = deconv_prim->stride;
                auto pad = deconv_prim->pad;
                ov::Strides dilation(spatial_rank, 1);
                auto output_padding = deconv_prim->output_paddings[0];
                auto grouped_weights_shape = deconv_prim->grouped_weights_shape;

                // remove deconvolution node and its connections to weights and biases, rename it and move to the optimized list
                p.remove_connection(input_node, deconv_node);
                std::vector<std::shared_ptr<program_node>> weight_connections;
                for (auto& weights_id : weights_nodes_id) {
                    auto weights_iter = p.nodes_map.find(weights_id);
                    if (weights_iter == p.nodes_map.end())
                        continue;

                    auto weights_node_ptr = weights_iter->second;
                    weight_connections.push_back(weights_node_ptr);
                    p.remove_connection(*weights_node_ptr, deconv_node);
                }

                ov::CoordinateDiff pad_begin(spatial_rank, 0);
                ov::CoordinateDiff pad_end(spatial_rank, 0);

                for (size_t i = 0; i < spatial_rank; i++) {
                    auto fs = filter_layout.spatial(spatial_rank - i - 1);
                    auto out_dim = output_pshape[2 + i].get_length();
                    auto in_dim = input_pshape[2 + i].get_length();

                    pad_begin[i] = (fs - 1) - std::abs(pad[i]);
                    pad_end[i] = (out_dim - 1) * stride[i] + fs - in_dim - pad_begin[i];
                }

                std::vector<std::shared_ptr<program_node>> bias_connections;
                for (auto& bias_id : biases_nodes_id) {
                    auto bias_iter = p.nodes_map.find(bias_id);
                    if (bias_iter == p.nodes_map.end())
                        continue;

                    auto bias_id_node_ptr = bias_iter->second;
                    bias_connections.push_back(bias_id_node_ptr);
                    p.remove_connection(*bias_id_node_ptr, deconv_node);
                }
                auto was_output = deconv_node.is_output();
                if (was_output) {
                    deconv_node.set_output(false);
                    auto& outputs = p.get_outputs();
                    outputs.erase(std::remove(outputs.begin(), outputs.end(), node.get()), outputs.end());
                }
                auto rename_id = deconv_node_id + "_tmp";
                p.rename(deconv_node, rename_id);

                // create convolution primitive
                auto conv_prim = std::make_shared<convolution>(deconv_node_id,
                                                               input_node_id,
                                                               weights_nodes_id[0],
                                                               biases_nodes_id.empty() ? "" : biases_nodes_id[0],
                                                               groups,
                                                               stride,
                                                               dilation,
                                                               pad_begin,
                                                               pad_end,
                                                               grouped_weights_shape,
                                                               ov::op::PadType::EXPLICIT);
                conv_prim->transposed = true;
                conv_prim->output_paddings = { output_padding };
                program_node& new_node = p.get_or_create(conv_prim);

                auto& conv_node = new_node.as<convolution>();
                conv_node.set_forced_impl_type(deconv_node.get_forced_impl_type());

                // add connections input->convolution, weights->convolution and bias->convolution
                p.add_connection(input_node, conv_node);

                for (auto& weight_node : weight_connections) {
                    p.add_connection(*weight_node, conv_node);
                }

                for (auto& bias_node : bias_connections) {
                    p.add_connection(*bias_node, conv_node);
                }

                auto deconv_node_itr = p.nodes_map.find(rename_id);
                if (deconv_node_itr != p.nodes_map.end()) {
                    auto deconv_node_ptr = deconv_node_itr->second;
                    p.replace_all_usages(*deconv_node_ptr, conv_node);
                    p.optimized_out.push_back(rename_id);
                    p.nodes_map.erase(rename_id);
                }

                if (was_output) {
                    conv_node.set_output(true);
                    p.get_outputs().push_back(&conv_node);
                }

                p.mark_if_data_flow(conv_node);
                conv_node.recalc_output_layout(true);

                update_processing_order = true;
            // current optimization only available for specific deconvolution parameters
            } else if (deconv_node.is_output() == false &&
               deconv_node.get_output_layout().feature() == 1 &&
               deconv_prim->stride[deconv_prim->stride.size() - 1] == 2 && deconv_prim->stride[deconv_prim->stride.size() - 2] == 2 &&
               filter_layout.spatial(0) == 9 && filter_layout.spatial(1) == 9 &&
               deconv_prim->pad[deconv_prim->pad.size() - 1] == 4 && deconv_prim->pad[deconv_prim->pad.size() - 2]  == 4 &&
               weights_nodes_id.size() == 1 && biases_nodes_id.size() == 1 &&
               input_node.get_output_layout().format == format::bfyx) {
                const auto scale_factor = deconv_prim->stride[deconv_prim->stride.size() - 1];
                auto spatial_rank = deconv_node.get_output_layout().get_spatial_rank();

                const auto& weight_node_id = weights_nodes_id.front();
                auto weights_node_ptr = p.nodes_map.find(weight_node_id)->second;
                const auto& weights_layout = weights_node_ptr->get_output_layout();
                const auto& weights_data_type = weights_layout.data_type;

                const auto& bias_node_id = biases_nodes_id.front();
                auto bias_id_node_ptr = p.nodes_map.find(bias_node_id)->second;
                const auto bias_data_type = bias_id_node_ptr->get_output_layout().data_type;

                // enable only for fp32 and fp16
                if (weights_data_type != data_types::f16 &&
                    weights_data_type != data_types::f32 &&
                    bias_data_type != data_types::f16 &&
                    bias_data_type != data_types::f32)
                    continue;

                // setting convolution parameters based on deconvolution params
                ov::Strides stride(spatial_rank, 1);
                ov::CoordinateDiff pad(spatial_rank, scale_factor);
                ov::Strides dilation(spatial_rank, 1);
                auto output_padding = deconv_prim->output_paddings[0];
                auto grouped_weights_shape = deconv_prim->grouped_weights_shape;

                // remove deconvolution node and its connections to weights and biases,
                // rename it and move to the optimized list
                p.remove_connection(input_node, deconv_node);

                p.remove_connection(*weights_node_ptr, deconv_node);
                p.remove_connection(*bias_id_node_ptr, deconv_node);

                auto rename_id = deconv_node_id + "_tmp";
                p.rename(deconv_node, rename_id);

                // reshape weights
                auto pixel_shuffle_size = static_cast<tensor::value_type>(scale_factor * scale_factor);
                int kernel_size = 5;
                tensor target_weights_size = { pixel_shuffle_size, filter_layout.feature(), kernel_size, kernel_size };
                auto target_weights_layout = layout{ weights_layout.data_type, weights_layout.format, target_weights_size };

                const primitive_id weight_replace_node_id = weight_node_id + "_conv_rpl";
                {
                     memory::ptr data_to_allocate = p.get_engine().allocate_memory(target_weights_layout);

                     std::vector<float> weights_vec_float;

                     if (weights_data_type == data_types::f16) {
                         mem_lock<ov::float16, mem_lock_type::read> src{ weights_node_ptr->as<data>().get_attached_memory_ptr(), stream };
                         for (uint32_t i = 0; i < weights_layout.count(); i++)
                             weights_vec_float.push_back(static_cast<float>(src.data()[i]));
                     } else {
                         mem_lock<float, mem_lock_type::read> src{ weights_node_ptr->as<data>().get_attached_memory_ptr(), stream };
                         for (uint32_t i = 0; i < weights_layout.count(); i++)
                             weights_vec_float.push_back(src.data()[i]);
                     }

                     std::vector<std::vector<std::vector<float> > > subpixel_weights(pixel_shuffle_size);

                     program_helpers::reshape_deconvolution_weights(weights_vec_float,
                         static_cast<int>(filter_layout.feature()),
                         static_cast<int>(filter_layout.spatial(0)),
                         static_cast<int>(filter_layout.spatial(1)),
                         static_cast<int>(scale_factor),
                         subpixel_weights);

                     if (weights_data_type == data_types::f16) {
                         mem_lock<ov::float16, mem_lock_type::write> dst{ data_to_allocate, stream};
                         program_helpers::set_weights_values<ov::float16>(dst.data(), subpixel_weights);
                     } else if (weights_data_type == data_types::f32) {
                         mem_lock<float, mem_lock_type::write> dst{ data_to_allocate, stream };
                         program_helpers::set_weights_values<float>(dst.data(), subpixel_weights);
                     } else {
                         throw std::logic_error("Not supported data type.");
                     }

                     auto data_node_weights_replace = std::make_shared<data>(weight_replace_node_id, data_to_allocate);
                     program_node& weights_replace_node = p.get_or_create(data_node_weights_replace);
                     auto& data_node = weights_replace_node.as<data>();
                     data_node.set_output_layout(target_weights_layout, false);
                }

                auto deconv_id_conv = deconv_node_id + "_conv";

                // create convolution primitive
                auto conv_prim = std::make_shared<convolution>(deconv_id_conv,
                                                               input_node_id,
                                                               weight_replace_node_id,
                                                               "",
                                                               1,
                                                               stride,
                                                               dilation,
                                                               pad,
                                                               pad,
                                                               grouped_weights_shape,
                                                               ov::op::PadType::EXPLICIT);
                conv_prim->output_paddings = {output_padding};
                program_node& created_node = p.get_or_create(conv_prim);

                auto& conv_node = created_node.as<convolution>();

                // add connections input->convolution, weights->convolution and bias->convolution
                p.add_connection(input_node, conv_node);

                {
                    auto weights_node_conv_rpl_ptr = p.nodes_map.find(weight_replace_node_id)->second;
                    p.add_connection(*weights_node_conv_rpl_ptr, conv_node);
                    p.inputs.push_back(weights_node_conv_rpl_ptr.get());
                }

                float bias = 0;

                if (bias_data_type == data_types::f16) {
                    mem_lock<ov::float16, mem_lock_type::read> src{ bias_id_node_ptr->as<data>().get_attached_memory_ptr(), stream };
                    bias = static_cast<float>(src.data()[0]);
                } else {
                    mem_lock<float, mem_lock_type::read> src{ bias_id_node_ptr->as<data>().get_attached_memory_ptr(), stream };
                    bias = src.data()[0];
                }
                auto pixel_shuffle_prim = std::make_shared<depth_to_space>(deconv_node_id, deconv_id_conv, 2, depth_to_space_mode::blocks_first);

                program_node& pixel_shuffle_node = p.get_or_create(pixel_shuffle_prim);
                auto bias_id = deconv_node_id + "_bias";
                auto bias_prim = std::make_shared<activation>(bias_id,
                                                              input_info(deconv_node_id),
                                                              activation_func::linear,
                                                              activation_additional_params{ 1, bias });
                program_node& bias_node = p.get_or_create(bias_prim);

                // add connections input->depth_to_space, depth_to_space->bias
                p.add_connection(conv_node, pixel_shuffle_node);
                p.add_connection(pixel_shuffle_node, bias_node);

                auto deconv_node_ptr = p.nodes_map.find(rename_id);
                if (deconv_node_ptr != p.nodes_map.end()) {
                    p.replace_all_usages(*deconv_node_ptr->second, bias_node);
                    p.optimized_out.push_back(rename_id);
                    p.nodes_map.erase(rename_id);
                }
                p.mark_if_data_flow(conv_node);
                conv_node.recalc_output_layout(true);

                update_processing_order = true;
            }
        }
    }

    if (update_processing_order) {
        p.get_processing_order().calc_processing_order(p);
    }
}
