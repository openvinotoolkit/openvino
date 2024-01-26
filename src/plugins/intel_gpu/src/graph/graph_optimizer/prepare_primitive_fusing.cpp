// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "program_helpers.h"
#include "pass_manager.h"

#include "pooling_inst.h"
#include "proposal_inst.h"
#include "roi_pooling_inst.h"
#include "quantize_inst.h"
#include "activation_inst.h"
#include "batch_to_space_inst.h"
#include "crop_inst.h"
#include "eltwise_inst.h"
#include "gemm_inst.h"
#include "lrn_inst.h"
#include "mutable_data_inst.h"
#include "mvn_inst.h"
#include "pooling_inst.h"
#include "normalize_inst.h"
#include "permute_inst.h"
#include "reshape_inst.h"
#include "softmax_inst.h"
#include "resample_inst.h"
#include "depth_to_space_inst.h"
#include "fully_connected_inst.h"
#include "space_to_depth_inst.h"
#include "gather_inst.h"
#include "gather_nd_inst.h"
#include "gather_elements_inst.h"
#include "scatter_update_inst.h"
#include "scatter_nd_update_inst.h"
#include "scatter_elements_update_inst.h"
#include "reverse_sequence_inst.h"
#include "shuffle_channels_inst.h"
#include "space_to_batch_inst.h"
#include "strided_slice_inst.h"
#include "cum_sum_inst.h"
#include "embedding_bag_inst.h"
#include "extract_image_patches_inst.h"
#include "reduce_inst.h"
#include <vector>
#include <map>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <deque>
#ifdef ENABLE_ONEDNN_FOR_GPU
#include <impls/onednn/utils.hpp>
#endif

using namespace cldnn;

void prepare_primitive_fusing::run(program& p) {
    fuse_reorders(p);
    remove_redundant_reshape(p);
    fuse_bias(p);
    fuse_simple_primitives(p);
    fuse_constant_transposes(p);
    optimize_fused_ops(p);
}

void prepare_primitive_fusing::remove_redundant_reshape(program &p) {
    auto node_itr = p.get_processing_order().begin();
    while (node_itr != p.get_processing_order().end()) {
        auto node = (*node_itr++);
        program_helpers::do_for_types<reshape>(*node, [&p](reshape_node& node) {
            for (const auto& prev : node.get_dependencies()) {
                if (!prev.first->is_type<reshape>())
                    return;
                if (prev.first->get_users().size() > 1 || prev.first->get_dependencies().size() > 1)
                    return;
                if (prev.first->as<reshape>().get_input_layout() == node.get_output_layout()) {
                    p.add_optimized_primitive_info(prev.first->id());
                    p.add_optimized_primitive_info(node.id());
                    p.extract_and_remove(*prev.first);
                    p.extract_and_remove(node);
                }
            }
        });
    }

    node_itr = p.get_processing_order().begin();
    while (node_itr != p.get_processing_order().end()) {
        auto node = (*node_itr++);
        program_helpers::do_for_types<reshape>(*node, [&p](reshape_node& node) {
            auto input_lay = node.get_input_layout();
            auto output_lay = node.get_output_layout();

            if (!node.is_in_place())
                return;

            if (input_lay.identical(output_lay)) {
                p.add_optimized_primitive_info(node.id());
                p.extract_and_remove(node);
            }
        });
    }

    node_itr = p.get_processing_order().begin();
    while (node_itr != p.get_processing_order().end()) {
        auto node = (*node_itr++);
        program_helpers::do_for_types<reorder>(*node, [&p](reorder_node& node) {
            auto& input_node = node.input();
            if (input_node.get_users().size() > 1 || node.get_users().size() > 1 || node.is_endpoint() || input_node.is_input())
                return;
            auto input_lay = input_node.get_output_layout();
            auto output_lay = node.get_output_layout();
            auto user_node = *node.get_users().begin();
            if (input_lay.identical(output_lay)) {
                if (node.has_mean() || !node.get_primitive()->subtract_per_feature.empty()) {
                    return;
                }
                if (!node.get_users().empty() && user_node->is_type<reshape>()) {
                    return;
                }
                p.add_optimized_primitive_info(node.id());
                p.extract_and_remove(node);
            }
        });
    }
}

void prepare_primitive_fusing::fuse_reorders(program &p) {
    // This loop tries fusing several reorders one by one (if present) into one reorder
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        if (node->is_output())
            continue;

        program_helpers::do_for_types<reorder>(*node, [&p](reorder_node& node) {
            auto& input = node.input();

            // Restrictions:
            // - inputs cannot be padded
            // - primitives input cannot be output
            // - input was optimized
            if (node.has_padded_dependency() || input.is_output() ||
                node.get_dependencies().size() != 1 || input.can_be_optimized())
                return;

            // - check if previous node is reorder with 1 user (and if the layouts are the same - remove reorder)
            // - do not fuse if current node has mean subtract
            if (input.get_users().size() != 1 || !input.is_type<reorder>() ||
                input.get_output_layout() != node.get_output_layout() || node.has_mean() ||
                !node.get_primitive()->subtract_per_feature.empty() ||
                node.get_primitive()->has_surface_input())
                return;

            p.add_optimized_primitive_info(node.id());

            auto output_layout = node.get_output_layout();
            input.set_output_layout(output_layout, false);
            p.extract_and_remove(node);
        });
    }
}

void prepare_primitive_fusing::fuse_bias(program &p) {
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        if (node->is_output() || node->is_constant() || !node->is_type<eltwise>())
            continue;

        auto& eltw_node = node->as<eltwise>();
        auto get_eltw_const_dep_idx = [](typed_program_node<eltwise>& eltw_node) {
            for (auto i = 0; i < static_cast<int32_t>(eltw_node.get_dependencies().size()); ++i) {
                if (eltw_node.get_dependency(i).is_constant())
                    return i;
            }
            return -1;
        };
        auto const_dep_idx = get_eltw_const_dep_idx(eltw_node);
        auto non_const_dep_idx = 1 - const_dep_idx;

        bool is_bias_add = eltw_node.get_primitive()->mode == eltwise_mode::sum &&
                           eltw_node.get_dependencies().size() == 2 &&
                           const_dep_idx >= 0 && const_dep_idx < 2;

        if (!is_bias_add)
            continue;

        auto is_3d_fully_connected = [](program_node& node) {
            if (!node.is_type<fully_connected>())
                return false;

            return node.as<fully_connected>().get_primitive()->input_size == 3;
        };


        if (node->get_output_layout().is_dynamic()) {
            auto broadcast_type = eltw_node.get_primitive()->broadcast_spec.m_type;
            if (!eltw_node.get_dependency(non_const_dep_idx).is_type<fully_connected>())
                continue;
            if (broadcast_type != ov::op::AutoBroadcastType::NUMPY && broadcast_type != ov::op::AutoBroadcastType::NONE)
                continue;
            // Numpy broadcast rule requires the dimension size which is not one to be same as the corresponding dimension of the other operand.
            // So we can ensure that the feature size is same for this broadcasting rule, thereby being considered as bias.
            auto const_shape = eltw_node.get_dependency(const_dep_idx).get_output_layout().get_shape();
            int32_t count_elements_not_one = 0;
            int32_t idx_element_not_one = -1;
            for (size_t i = 0; i < const_shape.size(); ++i) {
                if (const_shape[i] != 1) {
                    count_elements_not_one++;
                    idx_element_not_one = static_cast<int32_t>(i);
                }
                if (count_elements_not_one > 1)
                    break;
            }
            if (count_elements_not_one != 1 ||
                (idx_element_not_one != (static_cast<int32_t>(const_shape.size()) - 1))) {
                continue;
            }
        } else {
            cldnn::tensor::value_type out_features = node->get_output_layout().feature();
            bool is_3d_fc = false;

            // Change out_features value to proper dimension for 3D FC case
            if (is_3d_fully_connected(node->get_dependency(0))) {
                out_features = node->get_input_layout(0).spatial(1);
                is_3d_fc = true;
            } else if (is_3d_fully_connected(node->get_dependency(1))) {
                out_features = node->get_input_layout(1).spatial(1);
                is_3d_fc = true;
            }
            auto& const_dep = eltw_node.get_dependency(const_dep_idx);
            if ((const_dep.get_output_layout().feature() != out_features && !is_3d_fc) ||
                const_dep.get_output_layout().count() != static_cast<size_t>(out_features)) {
                continue;
            }
        }
        auto& bias_node = eltw_node.get_dependency(const_dep_idx);
        primitive_id bias_name = bias_node.id();
        auto& replace_candidate = eltw_node.get_dependency(non_const_dep_idx);

        if (bias_node.get_output_layout().data_type != replace_candidate.get_output_layout().data_type)
            continue;

        auto fuse_bias_f = [&p](program_node& prev_node, program_node& new_node, program_node& bias_node, program_node& eltw_node) {
            auto eltw_id = eltw_node.id();
            p.replace(prev_node, new_node);
            // Insert bias_node into 3-rd position in dependencies vector to get correct order in case of asymmetric quantization
            // which means that node can have > 2 dependencies even without bias
            auto port_idx = new_node.get_port_from_deps(bias_node.id());
            new_node.dependencies.insert(new_node.dependencies.begin() + 2, {&bias_node, port_idx});
            bias_node.users.push_back(&new_node);

            // Remove all edges connected with peer node
            while (eltw_node.get_dependencies().size() > 0) {
                auto& dep = eltw_node.get_dependency(eltw_node.get_dependencies().size() - 1);
                p.remove_connection(dep, eltw_node);
            }

            p.replace_all_usages(eltw_node, new_node);

            p.add_optimized_primitive_info(eltw_id, {new_node.id()});

            new_node.recalc_output_layout();
        };

        auto recalculate_biases = [&](data_node& original_node, data_node& second_node) -> bool {
            auto original_mem = original_node.get_attached_memory_ptr();
            auto second_mem = second_node.get_attached_memory_ptr();
            if (original_mem->count() != second_mem->count() || original_mem->get_layout().data_type != second_mem->get_layout().data_type)
                return false;

            switch (original_mem->get_layout().data_type) {
                case data_types::f32: {
                    cldnn::memory_ptr new_mem = p.get_engine().allocate_memory(original_mem->get_layout());
                    mem_lock<float, mem_lock_type::write> new_bias_mem(new_mem, p.get_stream());
                    mem_lock<float, mem_lock_type::read> original_bias_mem(original_mem, p.get_stream());
                    mem_lock<float, mem_lock_type::read> second_bias_mem(second_mem, p.get_stream());
                    float* original_data = original_bias_mem.data();
                    float* new_data = second_bias_mem.data();
                    for (size_t i = 0; i < original_bias_mem.size(); i++)
                        new_bias_mem[i] = original_data[i] + new_data[i];

                    original_node.attach_memory(new_mem);
                    break;
                }
                case data_types::f16: {
                    cldnn::memory_ptr new_mem = p.get_engine().allocate_memory(original_mem->get_layout());
                    mem_lock<ov::float16, mem_lock_type::write> new_bias_mem(new_mem, p.get_stream());
                    mem_lock<ov::float16, mem_lock_type::read> original_bias_mem(original_mem, p.get_stream());
                    mem_lock<ov::float16, mem_lock_type::read> second_bias_mem(second_mem, p.get_stream());
                    ov::float16* original_data = original_bias_mem.data();
                    ov::float16* new_data = second_bias_mem.data();
                    for (size_t i = 0; i < original_bias_mem.size(); i++) {
                        new_bias_mem[i] = original_data[i] + new_data[i];
                    }

                    original_node.attach_memory(new_mem);
                    break;
                }
                default:
                    return false;
            }
            return true;
        };

        if (replace_candidate.is_type<convolution>()) {
            auto& conv = replace_candidate.as<convolution>();
            auto desc = conv.get_primitive();
            primitive_id biases = bias_name;

            // If the primitive has biases, then we try to combine the values, or do nothing and keep as fused sum.
            if (conv.bias_term()) {
                if (conv.bias().is_type<data>() && bias_node.is_type<data>()) {
                    if (recalculate_biases(conv.bias().as<data>(), bias_node.as<data>())) {
                        p.replace_all_usages(eltw_node, conv);
                        p.add_optimized_primitive_info(eltw_node.id(), {conv.id()});
                        p.remove_all_connections(eltw_node);
                        p.remove_if_dangling(eltw_node);
                    }
                }
                continue;
            }

            auto conv_with_bias_prim = std::make_shared<convolution>(desc->id + "_tmp",
                                                                     desc->input[0],
                                                                     desc->weights,
                                                                     biases,
                                                                     desc->weights_zero_points,
                                                                     desc->activations_zero_points,
                                                                     desc->compensation,
                                                                     desc->groups,
                                                                     desc->stride,
                                                                     desc->dilation,
                                                                     desc->padding_begin,
                                                                     desc->padding_end,
                                                                     desc->grouped_weights_shape,
                                                                     conv.get_output_layout().data_type);

            // Copy transposed flag to new prim as convolution node might be produced by deconv -> conv replacement before this pass
            conv_with_bias_prim->transposed = desc->transposed;
            auto& new_conv_node = p.get_or_create(conv_with_bias_prim);

            fuse_bias_f(conv, new_conv_node, bias_node, eltw_node);
        } else if (replace_candidate.is_type<deconvolution>()) {
            auto& deconv = replace_candidate.as<deconvolution>();
            auto desc = deconv.get_primitive();
            std::vector<primitive_id> biases = {bias_name};

            // If the primitive has biases, then we try to combine the values, or do nothing and keep as fused sum.
            if (deconv.bias_term()) {
                if (deconv.bias().is_type<data>() && bias_node.is_type<data>()) {
                    if (recalculate_biases(deconv.bias().as<data>(), bias_node.as<data>())) {
                        p.replace_all_usages(eltw_node, deconv);
                        p.add_optimized_primitive_info(eltw_node.id(), {deconv.id()});
                        p.remove_all_connections(eltw_node);
                        p.remove_if_dangling(eltw_node);
                    }
                }
                continue;
            }

            auto deconv_with_bias_prim = std::make_shared<deconvolution>(desc->id + "_tmp",
                                                                         desc->input[0],
                                                                         desc->weights,
                                                                         biases,
                                                                         desc->groups,
                                                                         desc->stride,
                                                                         desc->pad,
                                                                         desc->dilations,
                                                                         deconv.get_output_layout().get_tensor(),
                                                                         desc->grouped_weights_shape);

            auto& new_deconv_node = p.get_or_create(deconv_with_bias_prim);
            fuse_bias_f(deconv, new_deconv_node, bias_node, eltw_node);
        } else if (replace_candidate.is_type<fully_connected>()) {
            auto& fc = replace_candidate.as<fully_connected>();
            auto desc = fc.get_primitive();

            // If the primitive has biases, then we try to combine the values, or do nothing and keep as fused sum.
            if (fc.bias_term()) {
                if (fc.bias().is_type<data>() && bias_node.is_type<data>()) {
                    if (recalculate_biases(fc.bias().as<data>(), bias_node.as<data>())) {
                        p.replace_all_usages(eltw_node, fc);
                        p.add_optimized_primitive_info(eltw_node.id(), {fc.id()});
                        p.remove_all_connections(eltw_node);
                        p.remove_if_dangling(eltw_node);
                    }
                }
                continue;
            }

            auto fc_with_bias_prim = std::make_shared<fully_connected>(desc->id + "_tmp",
                                                                       desc->input[0],
                                                                       desc->weights,
                                                                       bias_name,
                                                                       fc.get_output_layout().data_type,
                                                                       desc->output_paddings[0],
                                                                       desc->input_size);

            if (desc->compressed_weights) {
                fc_with_bias_prim->compressed_weights = true;
                fc_with_bias_prim->decompression_scale = desc->decompression_scale;
                fc_with_bias_prim->decompression_zero_point = desc->decompression_zero_point;
                if (desc->decompression_zero_point_scalar.has_value())
                    fc_with_bias_prim->decompression_zero_point_scalar = desc->decompression_zero_point_scalar.value();
            }
            auto& new_fc_node = p.get_or_create(fc_with_bias_prim);
            fuse_bias_f(fc, new_fc_node, bias_node, eltw_node);
        }
    }
}

void prepare_primitive_fusing::fuse_simple_primitives(program &p) {
    bool recalc_processing_order = false;
    std::map<primitive_id, std::vector<std::pair<primitive_id, size_t>>> fusing_history;

    const auto supports_immad = p.get_engine().get_device_info().supports_immad;
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        if (node->is_output() || node->is_constant())
            continue;

        auto is_grouped_conv = [](convolution_node& node) -> bool {
            auto in_layout = node.get_input_layout(0);
            return (node.get_groups() > 1 && node.get_groups() != static_cast<uint32_t>(in_layout.feature()));
        };

        auto conv_supports_fusings = [&](convolution_node& node) -> bool {
            if (_lo.get_optimization_attributes().use_onednn_impls == 1)
                return true;

            if (node.get_output_layout().is_dynamic() || node.get_input_layout().is_dynamic()) {
                return true;
            }

            if (node.get_primitive()->deformable_mode)
                return false;

            // Since reorder inputs is called after this pass
            // we have to check that blocked formats can be used in the network and layer is optimized for it.
            if ((node.get_output_layout().format == format::b_fs_yx_fsv16 ||
                _lo.should_select_b_fs_yx_fsv16_layout(node, node.get_input_layout(1))) &&
                 !is_grouped_conv(node))
                return true;

            if ((node.get_output_layout().format == format::bfzyx &&
                (!_lo.get_optimization_attributes().b_fs_zyx_fsv16_network || !_lo.is_format_optimized(node, format::b_fs_zyx_fsv16))))
                return true;

            if ((node.get_output_layout().format == format::fs_b_yx_fsv32 ||
                (_lo.get_optimization_attributes().fs_b_yx_fsv32_network &&
                 _lo.is_format_optimized(node, format::fs_b_yx_fsv32) && node.get_primitive()->groups == 1)))
                    return true;

            const size_t in_feature = node.get_input_layout(0).feature();
            if ((node.get_output_layout().format == format::b_fs_zyx_fsv16 ||
                 (_lo.is_format_optimized(node, format::b_fs_zyx_fsv16) &&
                  _lo.get_optimization_attributes().b_fs_zyx_fsv16_network)) && in_feature != 3)
                return true;

            if ((node.get_output_layout().format == format::bs_fs_yx_bsv16_fsv16 ||
                 (_lo.is_format_optimized(node, format::bs_fs_yx_bsv16_fsv16) &&
                  _lo.get_optimization_attributes().bs_fs_yx_bsv16_fsv16_network)) && node.get_primitive()->groups == 1)
                return true;

            if (node.get_output_layout().format == format::bs_fs_yx_bsv32_fsv32 || _lo.is_format_optimized(node, format::bs_fs_yx_bsv32_fsv32))
                return true;

            if (node.get_output_layout().format == format::bs_fs_yx_bsv32_fsv16 || _lo.is_format_optimized(node, format::bs_fs_yx_bsv32_fsv16))
                return true;

            auto in_dt = node.get_input_layout(0).data_type;

            // TODO: check if that's enough for correct work
            return data_type_traits::is_i8_u8(in_dt);
        };

        auto fc_supports_fusings = [&](fully_connected_node& node) -> bool {
            if (_lo.get_optimization_attributes().use_onednn_impls &&
                _lo.get_preferred_impl_type(node, format::any /*dummy*/) == impl_types::onednn) {
                return true;
            } else {
                auto in_dt = node.get_input_layout(0).data_type;
                return data_type_traits::is_i8_u8(in_dt);
            }
        };

        auto gemm_supports_fusings = [](gemm_node& node) -> bool {
            bool does_support_fusings = false;
            auto in0_dt = node.get_input_layout(0).data_type;
            auto in1_dt = node.get_input_layout(1).data_type;
            auto in0_fmt = node.get_input_layout(0).format;
            auto in1_fmt = node.get_input_layout(1).format;

            if (node.get_primitive()->indirect_a || node.get_primitive()->indirect_b)
                return false;

            if (data_type_traits::is_floating_point(in0_dt) &&
                data_type_traits::is_floating_point(in1_dt))
                does_support_fusings = true;

            if (data_type_traits::is_i8_u8(in0_dt) && in0_fmt == format::bfyx &&
                data_type_traits::is_i8_u8(in1_dt) && in1_fmt == format::bfyx) {
                if (node.get_inputs_count() == 3) {
                    auto in2_dt = node.get_input_layout(2).data_type;
                    auto in2_fmt = node.get_input_layout(2).format;
                    does_support_fusings = data_type_traits::is_i8_u8(in2_dt) && in2_fmt == format::bfyx ? true : false;
                } else {
                    does_support_fusings = true;
                }
            }

            auto gemm_prim = node.get_primitive();
            for (size_t idx = 0; idx < gemm_prim->output_order.size(); ++idx) {
                size_t output_order_idx = static_cast<size_t>(gemm_prim->output_order[idx]);
                if (idx != output_order_idx) {
                    does_support_fusings = false;
                    break;
                }
            }

            return does_support_fusings;
        };

        auto mvn_supports_fusings = [](mvn_node& node, bool for_eltwise = false) -> bool {
            auto in_layout = node.get_input_layout(0);
            if (node.get_primitive()->requires_alignment(in_layout.get_partial_shape()))
                return false;
            return data_type_traits::is_i8_u8(in_layout.data_type) || for_eltwise;
        };

        auto dts_supports_fusings = [](depth_to_space_node& node) -> bool {
            bool input_conv = node.get_dependency(0).is_type<convolution>();
            bool out_eltw = node.get_users().front()->is_type<eltwise>();
            if (input_conv && out_eltw) {
                auto& eltw = static_cast<const eltwise&>(*node.get_users().front()->get_primitive());
                auto& conv = node.get_dependency(0).as<convolution>();
                auto eltw_mode = eltw.mode == eltwise_mode::sum;
                auto conv_size = conv.get_input_layout(0).spatial(0) % 128 == 0 &&
                                 conv.get_input_layout(0).spatial(1) % 2 == 0;
                auto format = conv.get_output_layout().format == format::bfyx;
                auto dt = conv.get_output_layout().data_type == data_types::f16;
                if (eltw_mode && conv_size && format && dt)
                    return false;
            }

            return true;
        };

        auto reduce_supports_fusings = [&](reduce_node& node) -> bool {
            auto keep_dims = node.as<reduce>().get_primitive()->keep_dims;

            if (keep_dims)
                return true;

            return false;
        };

        auto eltwise_supports_fusings = [&](eltwise_node& node) -> bool {
            auto out_layout = node.get_output_layout();
            // Do not fuse if the estimated format is fs_b_yx_fsv32 because the optimized kernel does not support fusion
            if (out_layout.data_type == data_types::f16 && out_layout.is_static() && out_layout.batch() > 1 &&
                ((_lo.get_optimization_attributes().fs_b_yx_fsv32_network &&
                  !_lo.get_optimization_attributes().use_onednn_impls) ||
                 out_layout.format == format::fs_b_yx_fsv32)) {
                return false;
            }
            return true;
        };

        auto get_users_from_fusing_history = [&](const primitive_id& id) {
            std::vector<primitive_id> users;
            for (auto fusing_info : fusing_history) {
                auto key = fusing_info.first;
                auto dep_info_vec = fusing_info.second;
                auto iter = std::find_if(dep_info_vec.begin(), dep_info_vec.end(), [&](std::pair<primitive_id, size_t>& dep_info) {
                    return (id == dep_info.first);
                });
                if (iter != dep_info_vec.end()) {
                    users.push_back(key);
                }
            }
            return users;
        };

        auto input_data_supports_fusings = [&](cldnn::program_node& input_data, primitive_id current_node_id) -> bool {
            if (input_data.get_users().size() != 1) {
                // If input_data has fused primitives,
                // find original dependency of current_node using fusing_history
                // and check the number of users of it.
                // If the node has multiple users it's not fusible.
                if (!supports_immad && input_data.has_fused_primitives()) {
                    size_t num_original_dependencies = 0;
                    auto iter = fusing_history.find(current_node_id);
                    if (iter != fusing_history.end()) {
                        // Find current_node's original dependency list
                        for (auto& prim_id : iter->second) {
                            // find input_data's fused_prims in the prim_deps_ids
                            auto& fused_descs = input_data.get_fused_primitives();
                            auto origin_input_iter = std::find_if(fused_descs.begin(), fused_descs.end(),
                                                                    [&](cldnn::fused_primitive_desc& desc) {
                                return (desc.desc->id == prim_id.first);
                            });
                            if (origin_input_iter != fused_descs.end()) {
                                auto users = get_users_from_fusing_history(origin_input_iter->desc->id);
                                if (users.size() != 1) {
                                    return false;
                                }
                                num_original_dependencies++;
                            }
                        }
                    }
                    // If num_original_dependencies is zero, input_data is original parent
                    if (num_original_dependencies == 0) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            return true;
        };

        auto fuse_activation_f = [&](activation_node& activation_node) {
            auto activation_func = activation_node.get_primitive()->activation_function;
            if (supports_immad && activation_func == cldnn::activation_func::hyperbolic_tan) {
                return;
            }

            auto& input = activation_node.get_dependency(0);
            if (activation_node.get_dependencies().size() >= 3)
                return;

            if (!input_data_supports_fusings(input, activation_node.id()) || input.get_dependencies().empty())
                return;

            if (input.in_shape_of_subgraph || node->in_shape_of_subgraph)
                return;

            if (_lo.get_optimization_attributes().use_onednn_impls) {
                if (input.is_type<reshape>() || input.is_type<concatenation>())
                    return;
                auto additional_params_input = activation_node.get_primitive()->additional_params_input;
                if (activation_func == cldnn::activation_func::relu_negative_slope && !additional_params_input.empty() &&
                    (input.is_type<fully_connected>() || input.is_type<gemm>())) {
                    // prelu fusion is not implemented in oneDNN3.1 (CVS-108233)
                    return;
                }
                // Activation should not be fused if oneDNN does NOT support it
                if (_lo.is_primitive_implemented_for_onednn(input))  {
                    #ifdef ENABLE_ONEDNN_FOR_GPU
                    try {
                        onednn::convert_activation_func(activation_func);
                    } catch (...) {
                        return;
                    }
                    #endif
                }
            }

            bool should_fuse = input.is_type<convolution>() && conv_supports_fusings(input.as<convolution>());

            should_fuse |= input.is_type<fully_connected>() && fc_supports_fusings(input.as<fully_connected>());

            should_fuse |= input.is_type<gemm>() && gemm_supports_fusings(input.as<gemm>());

            should_fuse |= input.is_type<pooling>();

            should_fuse |= input.is_type<resample>();

            should_fuse |= input.is_type<mvn>();

            should_fuse |= input.is_type<normalize>() && data_type_traits::is_i8_u8(input.get_input_layout(0).data_type);

            should_fuse |= input.is_type<deconvolution>();

            should_fuse |= input.is_type<permute>();

            should_fuse |= input.is_type<activation>();

            should_fuse |= input.is_type<lrn>();

            should_fuse |= input.is_type<gather>();

            should_fuse |= input.is_type<gather_nd>();

            should_fuse |= input.is_type<gather_elements>();

            should_fuse |= input.is_type<scatter_update>();

            should_fuse |= input.is_type<scatter_nd_update>();

            should_fuse |= input.is_type<scatter_elements_update>();

            should_fuse |= input.is_type<depth_to_space>();

            should_fuse |= input.is_type<space_to_depth>();

            should_fuse |= input.is_type<batch_to_space>();

            should_fuse |= input.is_type<space_to_batch>();

            should_fuse |= input.is_type<reduce>() && reduce_supports_fusings(input.as<reduce>());

            should_fuse |= input.is_type<eltwise>() && eltwise_supports_fusings(input.as<eltwise>());

            should_fuse |= input.is_type<strided_slice>();

            bool legacy_fusion = activation_node.get_dependencies().size() == 1 &&
                                 !input.can_be_optimized() &&
                                 !activation_node.is_constant() &&
                                 !activation_node.has_fused_primitives() &&
                                 (input.is_type<concatenation>() ||
                                  input.is_type<convolution>() ||
                                  input.is_type<crop>() ||
                                  input.is_type<eltwise>() ||
                                  input.is_type<fully_connected>() ||
                                  input.is_type<normalize>() ||
                                  input.is_type<reorder>() ||
                                  (input.is_type<reshape>() && !input.is_dynamic()) ||
                                  input.is_type<roi_pooling>() ||
                                  input.is_type<softmax>() ||
                                  input.is_type<depth_to_space>() ||
                                  input.is_type<shuffle_channels>() ||
                                  input.is_type<strided_slice>() ||
                                  input.is_type<cum_sum>() ||
                                  input.is_type<reverse_sequence>() ||
                                  input.is_type<embedding_bag>() ||
                                  input.is_type<extract_image_patches>());

            if (!should_fuse && legacy_fusion) {
                GPU_DEBUG_LOG << activation_node.id() << " is fused by legacy conditions! Consider adding selected kernel with fused ops support\n";
            }

            should_fuse |= legacy_fusion;

            if (!should_fuse)
                return;

            // Onednn reorder does not support eltwise nor binary post operation
            if (_lo.get_optimization_attributes().use_onednn_impls && input.is_type<reorder>()) {
                return;
            }

            p.fuse_nodes(input, activation_node, &fusing_history);
        };

        auto fuse_quantize_f = [&](quantize_node& quantize_node) {
            auto& input_data = quantize_node.get_dependency(0);
            if (input_data.get_users().size() != 1 || input_data.get_dependencies().empty())
                return;

            if (input_data.in_shape_of_subgraph || node->in_shape_of_subgraph)
                return;

            auto out_layout = quantize_node.get_output_layout();
            auto in_layout = input_data.get_output_layout();
            if (in_layout.is_dynamic() || out_layout.is_dynamic())
                return;

            auto out_dt = out_layout.data_type;
            auto in_dt = input_data.get_input_layout(0).data_type;
            auto out_dt_is_i8_u8 = data_type_traits::is_i8_u8(out_dt);
            auto in_dt_is_i8_u8 = data_type_traits::is_i8_u8(in_dt);

            bool per_tensor_values = quantize_node.get_scale_shift_opt() &&
                                     quantize_node.get_per_tensor_input_scale() &&
                                     quantize_node.get_per_tensor_input_shift() &&
                                     quantize_node.get_per_tensor_input_range() &&
                                     quantize_node.get_per_tensor_output_scale() &&
                                     quantize_node.get_per_tensor_output_shift() &&
                                     quantize_node.get_per_tensor_output_range();

            bool should_fuse = input_data.is_type<convolution>() && conv_supports_fusings(input_data.as<convolution>()) &&
                           quantize_node.get_scale_shift_opt() &&
                           ((out_dt == data_types::f32 || out_dt == data_types::f16)  ||
                            in_layout.format == format::b_fs_yx_fsv16 ||
                            in_layout.format == format::bs_fs_yx_bsv32_fsv16 ||
                            (_lo.should_select_b_fs_yx_fsv16_layout(input_data.as<convolution>(), input_data.get_input_layout(1)) &&
                             !is_grouped_conv(input_data.as<convolution>())) ||
                           // Avoid fusing to b_fs_yx_fsv16 (and similar) kernels
                           _lo.get_optimization_attributes().use_onednn_impls ||
                           (in_dt_is_i8_u8 && out_dt_is_i8_u8));

            should_fuse |= input_data.is_type<pooling>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<fully_connected>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<lrn>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<gemm>() && gemm_supports_fusings(input_data.as<gemm>()) &&
                           quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<resample>() &&
                           quantize_node.get_scale_shift_opt() &&
                           out_dt_is_i8_u8;

            should_fuse |= input_data.is_type<mvn>() && mvn_supports_fusings(input_data.as<mvn>()) &&
                           quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<activation>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<normalize>() && quantize_node.get_scale_shift_opt() &&
                           in_dt_is_i8_u8;

            should_fuse |= input_data.is_type<deconvolution>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<gather>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<gather_nd>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<gather_elements>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<scatter_update>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<scatter_nd_update>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<scatter_elements_update>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<permute>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<depth_to_space>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<space_to_depth>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<batch_to_space>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<space_to_batch>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<reduce>() &&
                           reduce_supports_fusings(input_data.as<reduce>())
                           && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<eltwise>() && eltwise_supports_fusings(input_data.as<eltwise>()) && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<softmax>() &&
                           input_data.as<softmax>().get_primitive()->dimension == 1 &&
                           per_tensor_values;


            if (!should_fuse)
                return;

            p.fuse_nodes(input_data, quantize_node, &fusing_history);
        };

        auto fuse_eltwise_f = [&](eltwise_node& node) {
            std::shared_ptr<const cldnn::eltwise> prim = node.get_primitive();
            const std::vector<eltwise_mode> supported_modes = {
                eltwise_mode::sum,
                eltwise_mode::prod,
                eltwise_mode::sub,
                eltwise_mode::div
            };

            if (node.is_output() || node.get_inputs_count() != 2 ||
                std::find(supported_modes.begin(), supported_modes.end(), prim->mode) == supported_modes.end() ||
                !prim->stride.empty())
                return;

            std::vector<std::pair<program_node*, int32_t>> parents = node.get_dependencies();

            std::vector<bool> can_fuse_parents = { false, false };

            for (size_t i = 0; i < parents.size(); i++) {
                can_fuse_parents[i] = (parents[i].first->is_type<convolution>() &&
                                       conv_supports_fusings(parents[i].first->as<convolution>())) ||
                                      (parents[i].first->is_type<mvn>() &&
                                       mvn_supports_fusings(parents[i].first->as<mvn>(), true)) ||
                                      (parents[i].first->is_type<deconvolution>()) ||
                                      (parents[i].first->is_type<permute>()) ||
                                      (parents[i].first->is_type<resample>()) ||
                                      (parents[i].first->is_type<space_to_depth>()) ||
                                      (parents[i].first->is_type<fully_connected>() &&
                                       fc_supports_fusings(parents[i].first->as<fully_connected>())) ||
                                      (parents[i].first->is_type<gemm>() &&
                                       gemm_supports_fusings(parents[i].first->as<gemm>())) ||
                                      (parents[i].first->is_type<batch_to_space>()) ||
                                      (parents[i].first->is_type<space_to_batch>()) ||
                                      (parents[i].first->is_type<eltwise>() &&
                                       eltwise_supports_fusings(parents[i].first->as<eltwise>())) ||
                                      (parents[i].first->is_type<gather_nd>()) ||
                                      (parents[i].first->is_type<gather_elements>()) ||
                                      (parents[i].first->is_type<scatter_nd_update>()) ||
                                      (parents[i].first->is_type<scatter_elements_update>()) ||
                                      (parents[i].first->is_type<pooling>()) ||
                                      (parents[i].first->is_type<depth_to_space>() &&
                                       dts_supports_fusings(parents[i].first->as<depth_to_space>())) ||
                                      (parents[i].first->is_type<gather>()) ||
                                      (parents[i].first->is_type<reduce>() &&
                                       reduce_supports_fusings(parents[i].first->as<reduce>())) ||
                                      (parents[i].first->is_type<lrn>());
            }

            // Disable fusion to a node on constant path when second input is in data flow
            for (size_t i = 0; i < parents.size(); i++) {
                can_fuse_parents[i] = can_fuse_parents[i] && (!parents[i].first->is_constant() || parents[parents.size() - 1 - i].first->is_constant());
            }

            if (node.in_shape_of_subgraph || parents[0].first->in_shape_of_subgraph || parents[1].first->in_shape_of_subgraph)
                return;

            auto parent1 = parents[0];
            auto parent2 = parents[1];

            if (parent1.first->get_output_layout().is_static() && parent2.first->get_output_layout().is_static()) {
                auto p1_raw_size = parent1.first->get_output_layout().get_tensor().sizes();
                auto p2_raw_size = parent2.first->get_output_layout().get_tensor().sizes();
                for (unsigned k = 0; k < p1_raw_size.size(); k++) {
                    if (p1_raw_size[k] < p2_raw_size[k]) {
                        if (p1_raw_size[k] != 1)
                            return;
                        can_fuse_parents[0] = false;
                    } else if (p2_raw_size[k] < p1_raw_size[k]) {
                        if (p2_raw_size[k] != 1)
                            return;
                        can_fuse_parents[1] = false;
                    }
                }
            } else {
                // In case of dynamic shapes we check that parent & peer shapes are compatible to allow merge
                // This is required to avoid an issue when shape is partially defined and incorrectly propagated to further nodes
                // which may ruin shape inference
                // E.g. parent1 [?,?,768], parent2 [?,?,1]
                // expected eltw out shape: [?,?,768]
                // but w/o this check we can fuse eltwise to parent2 and return [?,?,1] as output shape which is unexpected
                auto parent1_pshape = parent1.first->get_output_pshape(0);
                auto parent2_pshape = parent2.first->get_output_pshape(0);
                auto out_pshape = node.get_output_pshape(0);

                auto are_compatible = [](const ov::PartialShape& out_shape, const ov::PartialShape& in_shape) -> bool {
                    if (out_shape.rank().get_length() != in_shape.rank().get_length())
                        return false;
                    bool compatible = true;
                    for (size_t i = 0; i < out_shape.size(); i++) {
                        auto& od = out_shape[i];
                        auto& id = in_shape[i];

                        if (od.is_static() && id.is_static()) {
                            compatible &= od.get_length() == id.get_length();
                        } else if (id.is_static()) {
                            compatible &= id.get_length() != 1;
                        }
                    }
                    return compatible;
                };

                can_fuse_parents[0] = can_fuse_parents[0] && are_compatible(out_pshape, parent1_pshape);
                can_fuse_parents[1] = can_fuse_parents[1] && are_compatible(out_pshape, parent2_pshape);
            }

            // We should have at least one node to fuse
            if (!can_fuse_parents[0] && !can_fuse_parents[1])
                return;

            // Choose node to fuse
            size_t fused_idx = can_fuse_parents[0] ? 0 : 1;
            size_t peer_idx  = can_fuse_parents[0] ? 1 : 0;

            int p1_pnum = p.get_processing_order().get_processing_number(parents[fused_idx].first);
            int p2_pnum = p.get_processing_order().get_processing_number(parents[peer_idx].first);

            auto p1_dt = parents[fused_idx].first->get_output_layout().data_type;
            auto p2_dt = parents[peer_idx].first->get_output_layout().data_type;

            if (can_fuse_parents[peer_idx] &&
               ((p1_pnum < p2_pnum && p1_dt == p2_dt) || (data_type_traits::is_floating_point(p2_dt) && !data_type_traits::is_floating_point(p1_dt)))) {
                // Swap in 2 cases:
                // 1. Both branches have same data type. Select branch with lower processing number
                // 2. Peer node has fp32 output type, but fused node - int8. In that case we have to fuse to the branch
                // with fp32 out type to avoid fp32 blobs in the quantized graph.
                std::swap(fused_idx, peer_idx);
            }

            auto fused_node = parents[fused_idx].first;
            auto peer_node = parents[peer_idx].first;

            if (_lo.get_optimization_attributes().use_onednn_impls) {
                auto eltw_in_size = peer_node->get_output_layout();
                if (eltw_in_size.is_dynamic())
                    return;
            }
            if (parent1.first->is_type<convolution>() && !conv_supports_fusings(parent1.first->as<convolution>()))
                return;

            if (parent2.first->is_type<convolution>() && !conv_supports_fusings(parent2.first->as<convolution>()))
                return;

            bool merge_allowed = true;
            // If fused node is not convolution and fused node has multiple users,
            //  follow the legacy checking rule
            if (!supports_immad && fused_node->is_type<convolution>() && fused_node->get_users().size() > 1) {
                // Allowed new pattern: Elt1, Act, Elt2, Elt3, Elt4 are fused to Conv1
                // * Conv1 -> Eltw1(Add) -> Act(Clamp) -> Eltw2(Mul) -> Eltw3(Mul) -> Eltw4(Add) -> Conv2
                // *   \–----------------------------------->/                          \---------> Eltw5(Div)
                //
                // Extended eltwise fusiblity checking rules
                //
                // 1. All fusing nodes should be eltwise or activation node
                // 2. All intermediate fusing nodes except last fusing node(i.e. Elt4) should have only eltwise or activation node as user.
                // 3. Currently eltwise and activations are allowed to be fused from multiple branches,
                //      but technically other fusable operations can be allowed too in the future.
                // 4. When node_queue has only one node, the while loop is ended and this node is fused to fused node(Conv1)
                //      node_queue having one node means all user nodes from fused node(Conv1) converge at that node.
                // 5. if node_queue has multiple nodes even if the level of current_node is max_levels, it cannot be fused.
                std::deque<std::pair<cldnn::program_node*, size_t>> node_queue; //std::pair<cldnn::program_node*, layer level>
                std::vector<cldnn::program_node*> node_history;
                node_queue.push_back(std::make_pair(fused_node, 0));

                const uint8_t max_levels = 5;
                do {
                    // Pop the current node from node_queue
                    // Add the current node to the node_history to verfiy the trace of checking
                    auto current_node = node_queue.front();
                    node_queue.pop_front();
                    if (std::find(node_history.begin(), node_history.end(), current_node.first) == node_history.end()) {
                        node_history.push_back(current_node.first);
                    }

                    if (current_node.second > max_levels) {
                        return;
                    }

                    // Push node to node_queue
                    // If the node is already existed in node_queue, do not add it to the node_queue.
                    auto push_node_queue = [&](cldnn::program_node* in_node, size_t level) {
                        auto iter = std::find_if(node_queue.begin(), node_queue.end(), [&](std::pair<cldnn::program_node*, size_t> element) {
                            return (in_node->id() == element.first->id());
                        });
                        if (iter == node_queue.end()) {
                            node_queue.push_back(std::make_pair(in_node, level));
                        }
                    };

                    // If the any user node is not eltwise(mul / add mode) and activation,
                    // the current node will be considered as last node and put it back into the node_queue
                    auto curr_users = current_node.first->get_users();
                    auto invalid_user_iter = std::find_if(curr_users.begin(), curr_users.end(), [&](cldnn::program_node* user) {
                        return (user->is_output() ||
                                    (!(user->is_type<eltwise>() && user->get_primitive()->input.size() == 2 &&
                                        (std::find(supported_modes.begin(), supported_modes.end(),
                                        (user->as<eltwise>()).get_primitive()->mode) != supported_modes.end())) &&
                                    !(user->is_type<activation>() && user->get_dependency(0).get_users().size() == 1)));
                    });

                    if (invalid_user_iter != curr_users.end()) {
                        // If fused_node(i.e. Conv1) have invalid user node(that is not activation and eltwise ndoe), it cannot be fused
                        if (fused_node->id() == current_node.first->id()) {
                            return;
                        }
                        push_node_queue(current_node.first, (current_node.second+1));
                        continue;
                    }

                    // Add user node in current node to the queue
                    // But, do not add the node that passed once, it is checked using node_history
                    for (auto& user : curr_users) {
                        auto iter = std::find(node_history.begin(), node_history.end(), user);
                        if (iter == node_history.end())
                            push_node_queue(user, current_node.second+1);
                    }
                } while (node_queue.size() > 1);
            } else {
                merge_allowed = fused_node->get_users().size() == 1;
                for (auto& parent : fused_node->get_dependencies())
                    if (parent.first->id() == peer_node->id())
                        merge_allowed = false;
            }

            if (!merge_allowed)
                return;

            if (p.get_processing_order().get_processing_number(fused_node) <
                p.get_processing_order().get_processing_number(peer_node))
                recalc_processing_order = true;

            // [WA]: Resample + Eltwise fusing causes accuracy issues without processing order update.
            // As in both cases processing order is valid, the issue might be connected with memory pool
            if (fused_node->is_type<resample>()) {
                recalc_processing_order = true;
            }

            p.fuse_nodes(*fused_node, node, &fusing_history);
        };

        program_helpers::do_for_types<activation, quantize, eltwise>(*node,
                fuse_activation_f,
                fuse_quantize_f,
                fuse_eltwise_f);
    }

    // Need to update processing order to handle cases when peer node processing number is greater
    // than fused node one
    if (recalc_processing_order)
        p.get_processing_order().calc_processing_order(p);
}

void prepare_primitive_fusing::fuse_constant_transposes(program& p) {
    std::function<const program_node*(const program_node*)> get_weightable_node =
        [&get_weightable_node](const program_node* node) -> const program_node* {
        if (node->get_users().empty())
            return nullptr;

        const auto* next_node = node->get_users().front();

        if (next_node->is_type<fully_connected>() ||
            next_node->is_type<deconvolution>() ||
            next_node->is_type<convolution>() ||
            next_node->is_type<deformable_conv>()) {
            size_t weights_offset = next_node->get_primitive()->input_size();
            std::vector<size_t> valid_weights_indices = {next_node->get_primitive()->input_size()};
            if (next_node->is_type<fully_connected>()) {
                auto& fc = next_node->as<fully_connected>();
                auto desc = fc.get_primitive();
                if (desc->compressed_weights) {
                    size_t scale_idx = weights_offset + (fc.bias_term() ? 2 : 1);
                    valid_weights_indices.push_back(scale_idx);
                    if (!desc->decompression_zero_point.empty()) {
                        valid_weights_indices.push_back(scale_idx + 1);
                    }
                }
            }

            for (auto& widx : valid_weights_indices) {
                if (&next_node->get_dependency(widx) == node) {
                    return next_node;
                }
            }
            return nullptr;
        }

        if (node->is_constant() && node->get_users().size() == 1)
            return get_weightable_node(next_node);

        return nullptr;
    };

    auto convert_data_format_by_order = [](format fmt, const std::vector<uint16_t>& order) -> format {
        const auto& old_order = fmt.dims_order();
        auto new_order = old_order;

        for (size_t i = 0; i < order.size(); ++i) {
            new_order[i] = old_order[order[i]];
        }

        return format::find_format(new_order, fmt.block_sizes());
    };

    std::vector<std::pair<program_node*, program_node*>> to_replace_nodes;

    auto& proc_order = p.get_processing_order();
    auto itr = proc_order.begin();
    while (itr != proc_order.end()) {
        auto& node = *itr++;

        if (!node->is_type<permute>())
            continue;

        auto& permute_node = node->as<permute>();

        auto weightable_node = get_weightable_node(&permute_node);

        if (weightable_node == nullptr || !permute_node.get_dependency(0).is_type<data>())
            continue;

        auto& prev_const = permute_node.get_dependency(0).as<data>();

        if (prev_const.get_users().size() != 1)
            continue;

        auto permute_order = permute_node.get_primitive()->permute_order;
        // Assumption that fc weights will be reshaped to 2d
        if (permute_order.size() != 2 && weightable_node->is_type<fully_connected>()) {
            if (permute_order == std::vector<uint16_t>{0, 2, 1} ||
                permute_order == std::vector<uint16_t>{0, 1, 3, 2}) {
                permute_order = {1, 0};
            } else {
                continue;
            }
        }

        format new_fmt = format::any;
        try {
            new_fmt = convert_data_format_by_order(prev_const.get_output_layout().format, permute_order);
        } catch(ov::Exception&) {
            continue;
        }

        layout updated_const_layout = prev_const.get_output_layout();
        updated_const_layout.format = new_fmt;
        updated_const_layout.set_partial_shape(permute_node.get_output_pshape());

        p.extract_and_remove(permute_node);

        const auto& new_mem = p.get_engine().reinterpret_buffer(prev_const.get_attached_memory(), updated_const_layout);

        auto new_const_prim = std::make_shared<data>(prev_const.id() + "_fused_with_transpose", new_mem);
        auto& new_const_node = p.get_or_create(new_const_prim);

        p.replace(prev_const, new_const_node);
        new_const_node.recalc_output_layout(false);

        // Add format reorder in case of onednn to avoid overhead during execution on weights memory allocation
        if (_lo.get_preferred_impl_type(const_cast<program_node&>(*weightable_node), format::any /*dummy*/) == impl_types::onednn) {
            auto next_node = new_const_node.get_users().front();
            bool can_be_fused = next_node->is_type<reorder>() &&
                                next_node->as<reorder>().is_simple_reorder() &&
                                next_node->get_users().size() == 1;
            if (can_be_fused) {
                layout reorder_layout = next_node->get_output_layout();
                reorder_layout.format = format::bfyx;

                auto new_reorder = std::make_shared<reorder>(next_node->id() + "_reorder_fmt", new_const_node.id(), reorder_layout);
                auto& new_reorder_node = p.get_or_create(new_reorder);
                to_replace_nodes.emplace_back(std::make_pair(next_node, &new_reorder_node));
            } else {
                layout reorder_layout = new_const_node.get_output_layout();
                reorder_layout.format = format::bfyx;

                auto new_reorder = std::make_shared<reorder>(new_const_node.id() + "_reorder_fmt", new_const_node.id(), reorder_layout);
                auto& new_reorder_node = p.get_or_create(std::move(new_reorder));
                p.add_intermediate(new_reorder_node, *new_const_node.get_users().front(), new_const_node);
                new_reorder_node.recalc_output_layout(false);
            }
        }
    }

    for (auto& nodes : to_replace_nodes) {
        p.replace(*nodes.first, *nodes.second);
        nodes.second->recalc_output_layout(false);
    }
}

void prepare_primitive_fusing::optimize_fused_ops(program& p) {
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        if (!node->has_fused_primitives())
            continue;

        // TODO: try more optimizations:
        // 1. clamp optimization
        // 2. fuse conv bias to quantize shift
        auto& fused_prims = node->get_fused_primitives();

        auto remove_deps_of_node = [&](cldnn::fused_primitive_desc& desc) {
            for (auto& prim : fused_prims) {
                if (desc.desc->id == prim.desc->id) {
                    continue;
                }
                auto rm_iter = prim.fused_deps.find(desc.desc->id);
                if (rm_iter != prim.fused_deps.end()) {
                    prim.fused_deps.erase(rm_iter);
                    prim.fused_deps.insert(desc.fused_deps.begin(), desc.fused_deps.end());
                }
            }
        };

        // Drop relu if the next fused op is quantize with u8 output and no in_shift
        auto fp_itr = fused_prims.begin();
        while (fp_itr != fused_prims.end()) {
            auto curr_itr = fp_itr++;
            if (fp_itr == fused_prims.end())
                break;

            auto& fp = *curr_itr;
            auto& fp_next = *fp_itr;
            if (fp.is_type<activation>() && fp_next.is_type<quantize>()) {
                const auto& act_prim = fp.typed_desc<activation>();;
                const auto& quant_param = fp_next.get_typed_fuse_params<QuantizeFuseParams>();

                bool can_skip = fp.deps.empty() && data_type_traits::is_i8_u8(fp_next.output_layout.data_type);
                can_skip &= ((act_prim->activation_function == activation_func::relu) && (act_prim->additional_params.a == 0.0f));
                can_skip &= (quant_param->_scale_shift_opt && !quant_param->_need_pre_shift);

                if (can_skip) {
                    remove_deps_of_node(fp);
                    fp_itr = fused_prims.erase(curr_itr);
                }
            }
        }
    }
}
