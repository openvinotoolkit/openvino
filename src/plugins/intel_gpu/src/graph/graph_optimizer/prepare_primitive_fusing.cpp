// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "program_helpers.h"
#include "pass_manager.h"

#include "pooling_inst.h"
#include "proposal_inst.h"
#include "roi_pooling_inst.h"
#include "quantize_inst.h"
#include "binary_convolution_inst.h"
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
#include "intel_gpu/runtime/error_handler.hpp"
#ifdef ENABLE_ONEDNN_FOR_GPU
#include <impls/onednn/utils.hpp>
#endif

void prepare_primitive_fusing::run(program& p) {
    fuse_reorders(p);
    remove_redundant_reshape(p);
    fuse_sigmoid_mul_to_swish(p);
    fuse_bias(p);
    fuse_simple_primitives(p);
    fuse_activations(p);
    optimize_fused_ops(p);
}

void prepare_primitive_fusing::remove_redundant_reshape(program &p) {
    auto node_itr = p.get_processing_order().begin();
    while (node_itr != p.get_processing_order().end()) {
        auto node = (*node_itr++);
        program_helpers::do_for_types<reshape>(*node, [&p](reshape_node& node) {
            for (auto prev : node.get_dependencies()) {
                if (!prev->is_type<reshape>())
                    return;
                if (prev->get_users().size() > 1)
                    return;
                if (prev->as<reshape>().input().get_output_layout() == node.get_output_layout()) {
                    p.add_optimized_primitive_info(prev->id());
                    p.add_optimized_primitive_info(node.id());
                    p.extract_and_remove(*prev);
                    p.extract_and_remove(node);
                }
            }
        });
    }

    node_itr = p.get_processing_order().begin();
    while (node_itr != p.get_processing_order().end()) {
        auto node = (*node_itr++);
        program_helpers::do_for_types<reshape>(*node, [&p](reshape_node& node) {
            auto input_lay = node.input().get_output_layout();
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

void prepare_primitive_fusing::fuse_sigmoid_mul_to_swish(program &p) {
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        if (node->is_output())
            continue;

        program_helpers::do_for_types<eltwise>(*node, [&p](eltwise_node& node) {
            if (node.get_dependencies().size() != 2)
                return;

            if (node.get_primitive()->mode != eltwise_mode::prod)
                return;

            auto& mul = node;
            program_node* activation_input = nullptr;
            size_t values_id = 1;
            if (node.get_dependency(0).is_type<activation>()) {
                activation_input = &node.get_dependency(0);
            } else if (node.get_dependency(1).is_type<activation>()) {
                activation_input = &node.get_dependency(1);
                values_id = 0;
            }

            if (!activation_input)
                return;

            if (activation_input->as<activation>().get_primitive()->activation_function != activation_func::logistic)
                return;

            auto& sigmoid = activation_input->as<activation>();

            if (sigmoid.is_output() || sigmoid.get_users().size() != 1)
                return;

            auto& input = node.get_dependency(values_id);

            if (&input != &sigmoid.input())
                return;

            activation_additional_params swish_params = {1.0f, 0.0f};
            auto swish_prim = std::make_shared<cldnn::activation>(mul.id() + "_swish", input.id(), activation_func::swish, swish_params);
            auto& swish = p.get_or_create(swish_prim);

            p.add_optimized_primitive_info(node.id(), {swish.id()});
            p.add_optimized_primitive_info(sigmoid.id(), {swish.id()});

            p.add_connection(input, swish);
            p.replace_all_usages(mul, swish);

            p.remove_all_connections(mul);
            p.remove_all_connections(sigmoid);

            p.remove_if_dangling(mul);
            p.remove_if_dangling(sigmoid);

            p.get_processing_order().insert_next(&input, &swish);

            swish.calc_output_layout();
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
                !node.get_primitive()->subtract_per_feature.empty())
                return;

            p.add_optimized_primitive_info(node.id());

            auto output_layout = node.get_output_layout();
            input.set_output_layout(output_layout, false);
            p.extract_and_remove(node);
        });
    }
}

void prepare_primitive_fusing::fuse_activations(program &p) {
    bool is_debug = p.get_options().get<build_option_type::debug>()->enabled();
    std::map<primitive_id, std::vector<std::pair<primitive_id, size_t>>> fusing_history;
    bool use_onednn_impls = false;

#ifdef ENABLE_ONEDNN_FOR_GPU
    auto& engine = p.get_engine();
    if (engine.get_device_info().supports_immad && engine.configuration().queue_type == queue_types::in_order)
        use_onednn_impls = true;
#endif

    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        program_helpers::do_for_types<activation>(*node, [&p, &is_debug, &fusing_history, &use_onednn_impls](activation_node& node) {
            auto& input = node.input();
            auto id = node.id();
            // Restrictions:
            // - inputs cannot be padded
            // - primitives input cannot be output
            // - no activation additional input
            // - input was optimized
            // - can't have fused primitives
            if (node.has_padded_dependency() || (input.is_output() && !is_debug) || node.is_output() ||
                node.get_dependencies().size() != 1 || input.can_be_optimized() || node.is_constant() ||
                node.has_fused_primitives())
                return;

            // - limit to primitives which implementations support activation fusing
            if (input.get_users().size() != 1 ||
                // TODO: new api needs to be created to read such caps
                // right now use whitelist so no new primitives will be affected in case of lack of fused activation
                // support
                (!input.is_type<concatenation>() && !input.is_type<convolution>() &&
                 !input.is_type<crop>() && !input.is_type<deconvolution>() && !input.is_type<eltwise>() &&
                 !input.is_type<fully_connected>() && !input.is_type<lrn>() && !input.is_type<normalize>() &&
                 !input.is_type<permute>() && !input.is_type<pooling>() && !input.is_type<reorder>() &&
                 !input.is_type<reshape>() && !input.is_type<roi_pooling>() &&
                 !input.is_type<softmax>() && !input.is_type<resample>() && !input.is_type<mvn>() &&
                 !input.is_type<depth_to_space>() && !input.is_type<batch_to_space>() &&
                 !input.is_type<space_to_batch>() && !input.is_type<gather>() && !input.is_type<scatter_update>() && !input.is_type<shuffle_channels>() &&
                 !input.is_type<scatter_nd_update>() &&
                 !input.is_type<gather_nd>() &&
                 !input.is_type<gather_elements>() &&
                 !input.is_type<strided_slice>() && !input.is_type<cum_sum>() && !input.is_type<reverse_sequence>() &&
                 !input.is_type<embedding_bag>() && !input.is_type<extract_image_patches>() &&
                 !input.is_type<activation>()))
                return;

            if (input.is_type<eltwise>()) {
                bool is_quantization = true;
                for (auto& in : input.get_dependencies()) {
                    if (!data_type_traits::is_i8_u8(in->get_output_layout().data_type))
                        is_quantization = false;
                }

                // TODO: Add new fused ops mechanism support to eltwise kernel in order to enable fusings in case of quantization
                if (is_quantization)
                    return;
            }

            if (use_onednn_impls) {
                if (input.is_type<reshape>() || input.is_type<concatenation>())
                    return;
                #ifdef ENABLE_ONEDNN_FOR_GPU
                // Activation should not be fused if it isn't supported in onednn
                try {
                    onednn::convert_activation_func(node.get_primitive()->activation_function);
                } catch (...) {
                    return;
                }
                #endif
            }

            if (input.get_fused_primitives().empty()) {
                input.add_fused_activation(node.get_primitive()->activation_function, node.get_primitive()->additional_params);
                for (size_t i = 0; i < node.get_fused_activations_funcs().size(); i++) {
                    input.add_fused_activation(node.get_fused_activations_funcs()[i],
                                               node.get_fused_activations_params()[i]);
                }
                auto outputPadding = node.get_output_layout().data_padding;
                input.set_output_padding(outputPadding);
                p.extract_and_remove(node);
            } else {
                // If node already has any fused node using new mechanism,
                // we can just use the same way and handle any amount of activations
                p.fuse_nodes(input, node, &fusing_history);
            }

            p.add_optimized_primitive_info(id, {input.id()});
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
        bool is_bias = eltw_node.get_primitive()->mode == eltwise_mode::sum &&
                       eltw_node.get_dependencies().size() == 2;

        if (!is_bias)
            continue;

        auto is_3d_fully_connected = [](program_node& node) {
            if (!node.is_type<fully_connected>())
                return false;

            return node.as<fully_connected>().get_primitive()->input_size == 3;
        };

        if (node->get_output_layout().is_dynamic())
            continue;

        size_t out_features = static_cast<size_t>(node->get_output_layout().feature());

        // Change out_features value to proper dimension for 3D FC case
        if (is_3d_fully_connected(node->get_dependency(0)))
            out_features = static_cast<size_t>(node->get_dependency(0).get_output_layout().spatial(1));
        else if (is_3d_fully_connected(node->get_dependency(1)))
            out_features = static_cast<size_t>(node->get_dependency(1).get_output_layout().spatial(1));

        int bias_idx = -1;
        for (size_t i = 0; i < eltw_node.get_dependencies().size(); i++) {
            auto& dep = eltw_node.get_dependency(i);
            if (dep.is_constant() && dep.get_output_layout().count() == out_features) {
                bias_idx = static_cast<int>(i);
                break;
            }
        }
        if (bias_idx < 0)
            continue;

        auto& bias_node = eltw_node.get_dependency(bias_idx);
        primitive_id bias_name = bias_node.id();
        auto& replace_candidate = bias_idx == 0 ? eltw_node.get_dependency(1) : eltw_node.get_dependency(0);

        if (bias_node.get_output_layout().data_type != replace_candidate.get_output_layout().data_type)
            continue;

        auto fuse_bias_f = [&p](program_node& prev_node, program_node& new_node, program_node& bias_node, program_node& eltw_node) {
            auto eltw_id = eltw_node.id();
            p.replace(prev_node, new_node);
            // Insert bias_node into 3-rd position in dependencies vector to get correct order in case of asymmetric quantization
            // which means that node can have > 2 dependencies even without bias
            new_node.dependencies.insert(new_node.dependencies.begin() + 2, &bias_node);
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

        auto recalculate_biases = [&](data_node& original_node, data_node& new_node) -> bool {
            auto original_mem = original_node.get_attached_memory_ptr();
            auto new_mem = new_node.get_attached_memory_ptr();
            if (original_mem->count() != new_mem->count() || original_mem->get_layout().data_type != new_mem->get_layout().data_type)
                return false;

            switch (original_mem->get_layout().data_type) {
                case data_types::f32: {
                    mem_lock<float, mem_lock_type::write> original_bias_mem(original_mem, p.get_stream());
                    mem_lock<float, mem_lock_type::read> new_bias_mem(new_mem, p.get_stream());
                    float* original_data = original_bias_mem.data();
                    float* new_data = new_bias_mem.data();
                    for (size_t i = 0; i < original_bias_mem.size(); i++)
                        original_data[i] += new_data[i];
                    break;
                }
                case data_types::f16: {
                    mem_lock<uint16_t, mem_lock_type::write> original_bias_mem(original_mem, p.get_stream());
                    mem_lock<uint16_t, mem_lock_type::read> new_bias_mem(new_mem, p.get_stream());
                    uint16_t* original_data = original_bias_mem.data();
                    uint16_t* new_data = new_bias_mem.data();
                    for (size_t i = 0; i < original_bias_mem.size(); i++) {
                        float new_val = half_to_float(original_data[i]) + half_to_float(new_data[i]);
                        original_data[i] = float_to_half(new_val);
                    }
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
            std::vector<primitive_id> biases = {bias_name};

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
                                                                     desc->groups,
                                                                     desc->stride,
                                                                     desc->pad,
                                                                     desc->dilation,
                                                                     conv.get_output_layout().get_tensor(),
                                                                     conv.get_output_layout().data_type,
                                                                     desc->grouped_weights_shape);

            conv_with_bias_prim->activations_zero_points = desc->activations_zero_points;
            conv_with_bias_prim->weights_zero_points = desc->weights_zero_points;
            conv_with_bias_prim->compensation = desc->compensation;
            auto& new_conv_node = p.get_or_create(conv_with_bias_prim);
            // Copy transposed flag to new prim as convolution node might be produced by deconv -> conv replacement before this pass
            new_conv_node.as<convolution>().set_transposed(conv.get_transposed());

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
                                                                       desc->output_padding,
                                                                       desc->input_size);

            auto& new_fc_node = p.get_or_create(fc_with_bias_prim);
            fuse_bias_f(fc, new_fc_node, bias_node, eltw_node);
        }
    }
}

void prepare_primitive_fusing::fuse_simple_primitives(program &p) {
    bool recalc_processing_order = false;
    std::map<primitive_id, std::vector<std::pair<primitive_id, size_t>>> fusing_history;

    const uint8_t supports_immad = p.get_engine().get_device_info().supports_immad;
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        if (node->is_output() || node->is_constant())
            continue;

        auto is_grouped_conv = [](convolution_node& node) -> bool {
            auto in_layout = node.get_dependency(0).get_output_layout();
            return (node.get_split() > 1 && node.get_split() != in_layout.feature()) ||
                   (node.get_groups() > 1 && node.get_groups() != static_cast<uint32_t>(in_layout.feature()));
        };

        auto conv_supports_fusings = [&](convolution_node& node) -> bool {
            if (_lo.get_optimization_attributes().use_onednn_impls == 1)
                return true;

            if (node.get_output_layout().is_dynamic()) {
                return true;
            }

            // Since reorder inputs is called after this pass
            // we have to check that blocked formats can be used in the network and layer is optimized for it.
            if ((node.get_output_layout().format == format::b_fs_yx_fsv16 ||
                _lo.should_select_b_fs_yx_fsv16_layout(node, node.get_dependency(1).get_output_layout())) &&
                 !is_grouped_conv(node))
                return true;

            if ((node.get_output_layout().format == format::bfzyx &&
                (!_lo.get_optimization_attributes().b_fs_zyx_fsv16_network || !_lo.is_format_optimized(node, format::b_fs_zyx_fsv16))))
                return true;

            if ((node.get_output_layout().format == format::fs_b_yx_fsv32 ||
                (_lo.get_optimization_attributes().fs_b_yx_fsv32_network &&
                 _lo.is_format_optimized(node, format::fs_b_yx_fsv32) && node.get_primitive()->groups == 1)))
                    return true;

            const size_t in_feature = node.get_dependency(0).get_output_layout().feature();
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

            auto in_dt = node.get_dependency(0).get_output_layout().data_type;

            // TODO: check if that's enough for correct work
            return data_type_traits::is_i8_u8(in_dt);
        };

        auto bin_conv_supports_eltw_fusings = [](binary_convolution_node& conv_node) -> bool {
            auto& eltw_node = static_cast<const eltwise_node&>(*conv_node.get_users().front());
            auto& eltw_prim = *eltw_node.get_primitive();

            if (eltw_node.get_dependencies().size() < 2)
                return false;

            auto const_layout = eltw_node.get_dependency(1).get_output_layout();
            auto conv_layout = conv_node.get_output_layout();
            auto per_channel_eltwise = const_layout.feature() == conv_layout.feature();

            if (eltw_node.get_dependency(1).is_constant() && per_channel_eltwise &&
                (eltw_prim.mode == eltwise_mode::sum || eltw_prim.mode == eltwise_mode::prod) &&
                all_ones(conv_node.get_primitive()->dilation))
                return true;

            return false;
        };

        auto fc_supports_fusings = [](fully_connected_node& node) -> bool {
            auto in_dt = node.get_dependency(0).get_output_layout().data_type;

            return data_type_traits::is_i8_u8(in_dt);
        };

        auto gemm_supports_fusings = [](gemm_node& node) -> bool {
            bool does_support_fusings = false;
            auto in0_dt = node.get_dependency(0).get_output_layout().data_type;
            auto in1_dt = node.get_dependency(1).get_output_layout().data_type;
            auto in0_fmt = node.get_dependency(0).get_output_layout().format;
            auto in1_fmt = node.get_dependency(1).get_output_layout().format;

            if (data_type_traits::is_floating_point(in0_dt) &&
                data_type_traits::is_floating_point(in1_dt))
                does_support_fusings = true;

            if (data_type_traits::is_i8_u8(in0_dt) && in0_fmt == format::bfyx &&
                data_type_traits::is_i8_u8(in1_dt) && in1_fmt == format::bfyx) {
                if (node.inputs_count() == 3) {
                    auto in2_dt = node.get_dependency(2).get_output_layout().data_type;
                    auto in2_fmt = node.get_dependency(2).get_output_layout().format;
                    does_support_fusings = data_type_traits::is_i8_u8(in2_dt) && in2_fmt == format::bfyx ? true : false;
                } else {
                    does_support_fusings = true;
                }
            }

            return does_support_fusings;
        };

        auto mvn_supports_fusings = [](mvn_node& node) -> bool {
            auto in_dt = node.get_dependency(0).get_output_layout().data_type;
            return data_type_traits::is_i8_u8(in_dt);
        };

        auto pooling_supports_fusings = [](pooling_node& node) -> bool {
            auto pooling_mode = node.get_primitive()->mode;
            return pooling_mode != cldnn::pooling_mode::max_with_argmax;
        };

        auto dts_supports_fusings = [](depth_to_space_node& node) -> bool {
            bool input_conv = node.get_dependency(0).is_type<convolution>();
            bool out_eltw = node.get_users().front()->is_type<eltwise>();
            if (input_conv && out_eltw) {
                auto& eltw = static_cast<const eltwise&>(*node.get_users().front()->get_primitive());
                auto& conv = node.get_dependency(0).as<convolution>();
                auto eltw_mode = eltw.mode == eltwise_mode::sum;
                auto conv_size = conv.get_dependency(0).get_output_layout().spatial(0) % 128 == 0 &&
                                 conv.get_dependency(0).get_output_layout().spatial(1) % 2 == 0;
                auto format = conv.get_output_layout().format == format::bfyx;
                auto dt = conv.get_output_layout().data_type == data_types::f16;
                if (eltw_mode && conv_size && format && dt)
                    return false;
            }

            return true;
        };

        auto reduce_supports_fusings = [&](reduce_node& node) -> bool {
            auto keep_dims = node.as<reduce>().get_primitive()->keep_dims;
            auto axes = node.as<reduce>().get_primitive()->axes;

            // If reduce tensor size is small, it sets not to fuse eltwise which leads to select oneDNN reference reduction
            // Because oneDNN optimized kernel does NOT support eltwise fusing
            if (p.get_engine().get_device_info().supports_immad && node.get_output_layout().get_dims().size() <= 4 &&
                ((find(axes.begin(), axes.end(), node.get_output_layout().get_rank() - 1) != axes.end() &&
                node.input().get_output_layout().spatial(0) > 16) ||
                (find(axes.begin(), axes.end(), node.get_output_layout().get_rank() - 2) != axes.end() &&
                node.input().get_output_layout().spatial(1) > 16) ||
                (find(axes.begin(), axes.end(), 1) != axes.end() &&
                node.input().get_output_layout().feature() > 16) ||
                (node.get_output_layout().count() > 256)))
                return false;

            if (keep_dims)
                return true;

            return false;
        };

        auto eltwise_supports_fusings = [&](eltwise_node& node) -> bool {
            auto out_layout = node.get_output_layout();
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
            auto& input_data = activation_node.get_dependency(0);
            if (activation_node.get_dependencies().size() >= 3)
                return;

            if (!input_data_supports_fusings(input_data, activation_node.id()) || input_data.get_dependencies().empty())
                return;

            if (_lo.get_optimization_attributes().use_onednn_impls) {
                #ifdef ENABLE_ONEDNN_FOR_GPU
                // Activation should not fused if it isn't supported in onednn
                try {
                    onednn::convert_activation_func(activation_node.get_primitive()->activation_function);
                } catch (...) {
                    return;
                }
                #endif
            }

            bool should_fuse = input_data.is_type<binary_convolution>();

            should_fuse |= input_data.is_type<convolution>() && conv_supports_fusings(input_data.as<convolution>());

            should_fuse |= input_data.is_type<fully_connected>() && fc_supports_fusings(input_data.as<fully_connected>());

            should_fuse |= input_data.is_type<gemm>() && gemm_supports_fusings(input_data.as<gemm>());

            should_fuse |= input_data.is_type<pooling>() && pooling_supports_fusings(input_data.as<pooling>());

            should_fuse |= input_data.is_type<resample>();

            should_fuse |= input_data.is_type<mvn>();

            should_fuse |= input_data.is_type<normalize>() && data_type_traits::is_i8_u8(input_data.get_dependency(0).get_output_layout().data_type);

            should_fuse |= input_data.is_type<deconvolution>();

            should_fuse |= input_data.is_type<permute>();

            should_fuse |= input_data.is_type<activation>();

            should_fuse |= input_data.is_type<lrn>();

            should_fuse |= input_data.is_type<gather>();

            should_fuse |= input_data.is_type<gather_nd>();

            should_fuse |= input_data.is_type<gather_elements>();

            should_fuse |= input_data.is_type<scatter_update>();

            should_fuse |= input_data.is_type<scatter_nd_update>();

            should_fuse |= input_data.is_type<scatter_elements_update>();

            should_fuse |= input_data.is_type<depth_to_space>();

            should_fuse |= input_data.is_type<space_to_depth>();

            should_fuse |= input_data.is_type<batch_to_space>();

            should_fuse |= input_data.is_type<space_to_batch>();

            should_fuse |= input_data.is_type<reduce>() && reduce_supports_fusings(input_data.as<reduce>());

            should_fuse |= input_data.is_type<eltwise>() && eltwise_supports_fusings(input_data.as<eltwise>());

            if (!should_fuse)
                return;

            p.fuse_nodes(input_data, activation_node, &fusing_history);
        };

        auto fuse_quantize_f = [&](quantize_node& quantize_node) {
            auto& input_data = quantize_node.get_dependency(0);
            if (input_data.get_users().size() != 1 || input_data.get_dependencies().empty())
                return;

            auto& input_lo = quantize_node.get_dependency(1);
            auto& input_hi = quantize_node.get_dependency(2);

            auto out_layout = quantize_node.get_output_layout();
            auto in_layout = input_data.get_output_layout();
            auto out_dt = out_layout.data_type;
            auto in_dt = input_data.get_dependency(0).get_output_layout().data_type;
            auto out_dt_is_i8_u8 = data_type_traits::is_i8_u8(out_dt);
            auto in_dt_is_i8_u8 = data_type_traits::is_i8_u8(in_dt);

            bool per_tensor_values = quantize_node.get_scale_shift_opt() &&
                                     quantize_node.get_per_tensor_input_scale() &&
                                     quantize_node.get_per_tensor_input_shift() &&
                                     quantize_node.get_per_tensor_input_range() &&
                                     quantize_node.get_per_tensor_output_scale() &&
                                     quantize_node.get_per_tensor_output_shift() &&
                                     quantize_node.get_per_tensor_output_range();

            bool should_fuse = input_data.is_type<binary_convolution>() &&
                               ((out_dt == data_types::bin &&
                               quantize_node.get_dependencies().size() == 5 &&
                               ((in_layout.feature() == input_lo.get_output_layout().feature() &&
                                 in_layout.feature() == input_hi.get_output_layout().feature()) ||
                                (input_lo.get_output_layout().feature() == 1 &&
                                 input_hi.get_output_layout().feature() == 1)))) &&
                                 all_ones(input_data.as<binary_convolution>().get_primitive()->dilation);

            should_fuse |= input_data.is_type<convolution>() && conv_supports_fusings(input_data.as<convolution>()) &&
                           quantize_node.get_scale_shift_opt() &&
                           ((out_dt == data_types::f32 || out_dt == data_types::f16)  ||
                            in_layout.format == format::b_fs_yx_fsv16 ||
                            in_layout.format == format::bs_fs_yx_bsv32_fsv16 ||
                            (_lo.should_select_b_fs_yx_fsv16_layout(input_data.as<convolution>(), input_data.get_dependency(1).get_output_layout()) &&
                             !is_grouped_conv(input_data.as<convolution>())) ||
                           // Avoid fusing to b_fs_yx_fsv16 (and similar) kernels
                           _lo.get_optimization_attributes().use_onednn_impls ||
                           (in_dt_is_i8_u8 && out_dt_is_i8_u8));

            should_fuse |= input_data.is_type<pooling>() && quantize_node.get_scale_shift_opt() &&
                           pooling_supports_fusings(input_data.as<pooling>());

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

            if (node.is_output() || node.inputs_count() != 2 ||
                std::find(supported_modes.begin(), supported_modes.end(), prim->mode) == supported_modes.end() ||
                !prim->stride.empty())
                return;

            std::vector<cldnn::program_node*> parents = node.get_dependencies();
            std::list<cldnn::program_node*> users = node.get_users();

            std::vector<bool> can_fuse_parents = { false, false };

            for (size_t i = 0; i < parents.size(); i++) {
                can_fuse_parents[i] = (parents[i]->is_type<convolution>() && conv_supports_fusings(parents[i]->as<convolution>())) ||
                                      (parents[i]->is_type<binary_convolution>() && bin_conv_supports_eltw_fusings(parents[i]->as<binary_convolution>())) ||
                                      (parents[i]->is_type<mvn>() && mvn_supports_fusings(parents[i]->as<mvn>())) ||
                                      (parents[i]->is_type<deconvolution>()) ||
                                      (parents[i]->is_type<permute>()) ||
                                      (parents[i]->is_type<resample>()) ||
                                      (parents[i]->is_type<space_to_depth>()) ||
                                      (parents[i]->is_type<fully_connected>() && fc_supports_fusings(parents[i]->as<fully_connected>())) ||
                                      (parents[i]->is_type<gemm>() && gemm_supports_fusings(parents[i]->as<gemm>())) ||
                                      (parents[i]->is_type<batch_to_space>()) ||
                                      (parents[i]->is_type<space_to_batch>()) ||
                                      (parents[i]->is_type<eltwise>() && eltwise_supports_fusings(parents[i]->as<eltwise>())) ||
                                      (parents[i]->is_type<gather_nd>()) ||
                                      (parents[i]->is_type<gather_elements>()) ||
                                      (parents[i]->is_type<scatter_nd_update>()) ||
                                      (parents[i]->is_type<scatter_elements_update>()) ||
                                      (parents[i]->is_type<pooling>() && pooling_supports_fusings(parents[i]->as<pooling>())) ||
                                      (parents[i]->is_type<depth_to_space>() && dts_supports_fusings(parents[i]->as<depth_to_space>())) ||
                                      (parents[i]->is_type<gather>()) ||
                                      (parents[i]->is_type<reduce>() && reduce_supports_fusings(parents[i]->as<reduce>())) ||
                                      (parents[i]->is_type<lrn>());
            }

            // Disable fusion to a node on constant path when second input is in data flow
            for (size_t i = 0; i < parents.size(); i++) {
                can_fuse_parents[i] = can_fuse_parents[i] && (!parents[i]->is_constant() || parents[parents.size() - 1 - i]->is_constant());
            }

            auto parent1 = parents[0];
            auto parent2 = parents[1];

            if (parent1->get_output_layout().is_static() && parent2->get_output_layout().is_static()) {
                auto p1_raw_size = parent1->get_output_layout().get_tensor().sizes();
                auto p2_raw_size = parent2->get_output_layout().get_tensor().sizes();
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
            }

            // We should have at least one node to fuse
            if (!can_fuse_parents[0] && !can_fuse_parents[1])
                return;

            // Choose node to fuse
            size_t fused_idx = can_fuse_parents[0] ? 0 : 1;
            size_t peer_idx  = can_fuse_parents[0] ? 1 : 0;

            int p1_pnum = p.get_processing_order().get_processing_number(parents[fused_idx]);
            int p2_pnum = p.get_processing_order().get_processing_number(parents[peer_idx]);

            auto p1_dt = parents[fused_idx]->get_output_layout().data_type;
            auto p2_dt = parents[peer_idx]->get_output_layout().data_type;

            if (can_fuse_parents[peer_idx] &&
               ((p1_pnum < p2_pnum && p1_dt == p2_dt) || (data_type_traits::is_floating_point(p2_dt) && !data_type_traits::is_floating_point(p1_dt)))) {
                // Swap in 2 cases:
                // 1. Both branches have same data type. Select branch with lower processing number
                // 2. Peer node has fp32 output type, but fused node - int8. In that case we have to fuse to the branch
                // with fp32 out type to avoid fp32 blobs in the quantized graph.
                std::swap(fused_idx, peer_idx);
            }

            auto fused_node = parents[fused_idx];
            auto peer_node = parents[peer_idx];

            if (_lo.get_optimization_attributes().use_onednn_impls) {
                auto eltw_in_size = peer_node->get_output_layout();
                if (eltw_in_size.is_dynamic())
                    return;
            }
            if (parent1->is_type<convolution>() && !conv_supports_fusings(parent1->as<convolution>()))
                return;

            if (parent2->is_type<convolution>() && !conv_supports_fusings(parent2->as<convolution>()))
                return;

            bool merge_allowed = true;
            // If fused node is not convolution and fused node has multiple users,
            //  follow the legacy checking rule
            if (!supports_immad && fused_node->is_type<convolution>() && fused_node->get_users().size() > 1) {
                // Allowed new pattern: Elt1, Act, Elt2, Elt3, Elt4 are fused to Conv1
                // * Conv1 -> Eltw1(Add) -> Act(Clamp) -> Eltw2(Mul) -> Eltw3(Mul) -> Eltw4(Add) -> Conv2
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
                    if (parent->id() == peer_node->id())
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
                const auto& quant_param = fp_next.get_typed_fuse_params<kernel_selector::quantize_fuse_params>();

                bool can_skip = fp.deps.empty() && data_type_traits::is_i8_u8(fp_next.output_layout.data_type);
                can_skip &= ((act_prim->activation_function == activation_func::relu) && (act_prim->additional_params.a == 0.0f));
                can_skip &= (quant_param->scale_shift_opt && !quant_param->has_pre_shift);

                if (can_skip) {
                    remove_deps_of_node(fp);
                    fp_itr = fused_prims.erase(curr_itr);
                }
            }
        }
    }
}
