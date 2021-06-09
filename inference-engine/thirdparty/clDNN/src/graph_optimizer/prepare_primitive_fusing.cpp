// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "api/pooling.hpp"
#include "api/proposal.hpp"
#include "api/roi_pooling.hpp"

#include "program_helpers.h"
#include "pass_manager.h"

#include "quantize_inst.h"
#include "binary_convolution_inst.h"
#include "activation_inst.h"
#include "batch_to_space_inst.h"
#include "crop_inst.h"
#include "eltwise_inst.h"
#include "fused_conv_eltwise_inst.h"
#include "gemm_inst.h"
#include "lrn_inst.h"
#include "mutable_data_inst.h"
#include "mvn_inst.h"
#include "pooling_inst.h"
#include "normalize_inst.h"
#include "permute_inst.h"
#include "reshape_inst.h"
#include "softmax_inst.h"
#include "scale_inst.h"
#include "resample_inst.h"
#include "depth_to_space_inst.h"
#include "space_to_depth_inst.h"
#include "gather_inst.h"
#include "gather_nd_inst.h"
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
#include "error_handler.h"

void prepare_primitive_fusing::run(program_impl& p) {
    fuse_reorders(p);
    fuse_sigmoid_mul_to_swish(p);
    fuse_bias(p);
    fuse_simple_primitives(p);
    fuse_activations(p);
    optimize_fused_ops(p);
}

void prepare_primitive_fusing::fuse_sigmoid_mul_to_swish(program_impl &p) {
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

void prepare_primitive_fusing::fuse_reorders(program_impl &p) {
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

void prepare_primitive_fusing::fuse_activations(program_impl &p) {
    bool is_debug = p.get_options().get<build_option_type::debug>()->enabled();
    std::map<primitive_id, std::vector<primitive_id>> fusing_history;
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        program_helpers::do_for_types<activation>(*node, [&p, &is_debug, &fusing_history](activation_node& node) {
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
                 !input.is_type<reshape>() && !input.is_type<roi_pooling>() && !input.is_type<scale>() &&
                 !input.is_type<softmax>() && !input.is_type<resample>() && !input.is_type<mvn>() &&
                 !input.is_type<depth_to_space>() && !input.is_type<batch_to_space>() &&
                 !input.is_type<space_to_batch>() && !input.is_type<gather>() && !input.is_type<scatter_update>() && !input.is_type<shuffle_channels>() &&
                 !input.is_type<scatter_nd_update>() &&
                 !input.is_type<gather_nd>() &&
                 !input.is_type<strided_slice>() && !input.is_type<cum_sum>() && !input.is_type<reverse_sequence>() &&
                 !input.is_type<embedding_bag>() && !input.is_type<extract_image_patches>() &&
                 !input.is_type<fused_conv_eltwise>() && !input.is_type<activation>()))
                return;

            if (input.is_type<eltwise>()) {
                bool is_quantization = true;
                for (auto& in : input.get_dependencies()) {
                    if (in->get_output_layout().data_type != data_types::u8 && in->get_output_layout().data_type != data_types::i8)
                        is_quantization = false;
                }

                // TODO: Add new fused ops mechanism support to eltwise kernel in order to enable fusings in case of quantization
                if (is_quantization)
                    return;
            }

            if (input.get_fused_primitives().empty()) {
                input.add_fused_activation(node.get_primitive()->activation_function, node.get_primitive()->additional_params);
                for (size_t i = 0; i < node.get_fused_activations_funcs().size(); i++) {
                    input.add_fused_activation(node.get_fused_activations_funcs()[i],
                                               node.get_fused_activations_params()[i]);
                }
                input.set_output_padding(node.get_output_layout().data_padding);
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

void prepare_primitive_fusing::fuse_bias(program_impl &p) {
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

        size_t out_features = static_cast<size_t>(node->get_output_layout().size.feature[0]);

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

        if (replace_candidate.is_type<convolution>()) {
            auto& conv = replace_candidate.as<convolution>();
            auto desc = conv.get_primitive();
            std::vector<primitive_id> biases = {bias_name};

            auto conv_with_bias_prim = std::make_shared<convolution>(desc->id + "_tmp",
                                                                     desc->input[0],
                                                                     desc->weights,
                                                                     biases,
                                                                     desc->groups,
                                                                     desc->stride,
                                                                     desc->input_offset,
                                                                     desc->dilation,
                                                                     conv.get_output_layout().size,
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

            auto deconv_with_bias_prim = std::make_shared<deconvolution>(desc->id + "_tmp",
                                                                         desc->input[0],
                                                                         desc->weights,
                                                                         biases,
                                                                         desc->groups,
                                                                         desc->stride,
                                                                         desc->input_offset,
                                                                         deconv.get_output_layout().size,
                                                                         desc->grouped_weights_shape);

            auto& new_deconv_node = p.get_or_create(deconv_with_bias_prim);
            fuse_bias_f(deconv, new_deconv_node, bias_node, eltw_node);
        } else if (replace_candidate.is_type<fully_connected>()) {
            auto& fc = replace_candidate.as<fully_connected>();
            auto desc = fc.get_primitive();
            auto fc_with_bias_prim = std::make_shared<fully_connected>(desc->id + "_tmp",
                                                                       desc->input[0],
                                                                       desc->weights,
                                                                       bias_name,
                                                                       fc.get_output_layout().data_type);

            auto& new_fc_node = p.get_or_create(fc_with_bias_prim);
            fuse_bias_f(fc, new_fc_node, bias_node, eltw_node);
        }
    }
}

void prepare_primitive_fusing::fuse_simple_primitives(program_impl &p) {
    bool recalc_processing_order = false;
    std::map<primitive_id, std::vector<primitive_id>> fusing_history;

    const uint8_t supports_immad = p.get_engine().get_device_info().supports_immad;
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        if (node->is_output() || node->is_constant())
            continue;

        auto is_grouped_conv = [](convolution_node& node) -> bool {
            auto in_size = node.get_dependency(0).get_output_layout().size;
            return (node.get_split() > 1 && node.get_split() != in_size.feature[0]) ||
                   (node.get_groups() > 1 && node.get_groups() != static_cast<uint32_t>(in_size.feature[0]));
        };

        auto conv_supports_fusings = [&](convolution_node& node) -> bool {
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

            const size_t in_feature = node.get_dependency(0).get_output_layout().size.feature[0];
            if ((node.get_output_layout().format == format::b_fs_zyx_fsv16 ||
                 (_lo.is_format_optimized(node, format::b_fs_zyx_fsv16) &&
                  _lo.get_optimization_attributes().b_fs_zyx_fsv16_network)) && in_feature != 3)
                return true;

            if ((node.get_output_layout().format == format::bs_fs_yx_bsv16_fsv16 ||
                 (_lo.is_format_optimized(node, format::bs_fs_yx_bsv16_fsv16) &&
                  _lo.get_optimization_attributes().bs_fs_yx_bsv16_fsv16_network)) && node.get_primitive()->groups == 1)
                return true;

            auto in_dt = node.get_dependency(0).get_output_layout().data_type;

            // TODO: check if that's enough for correct work
            if (in_dt == data_types::u8 || in_dt == data_types::i8)
                return true;

            return false;
        };

        auto bin_conv_supports_eltw_fusings = [](binary_convolution_node& conv_node) -> bool {
            auto& eltw_node = static_cast<const eltwise_node&>(*conv_node.get_users().front());
            auto& eltw_prim = *eltw_node.get_primitive();

            if (eltw_node.get_dependencies().size() < 2)
                return false;

            auto const_layout = eltw_node.get_dependency(1).get_output_layout();
            auto conv_layout = conv_node.get_output_layout();
            auto per_channel_eltwise = const_layout.size.feature[0] == conv_layout.size.feature[0];

            if (eltw_node.get_dependency(1).is_constant() && per_channel_eltwise &&
                (eltw_prim.mode == eltwise_mode::sum || eltw_prim.mode == eltwise_mode::prod) &&
                (conv_node.get_primitive()->dilation == tensor{1}))
                return true;

            return false;
        };

        auto fc_supports_fusings = [](fully_connected_node& node) -> bool {
            auto in_dt = node.get_dependency(0).get_output_layout().data_type;

            if (in_dt == data_types::u8 || in_dt == data_types::i8)
                return true;

            return false;
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

            if ((in0_dt == data_types::u8 || in0_dt == data_types::i8) &&
                (in1_dt == data_types::u8 || in1_dt == data_types::i8) &&
                in0_fmt == format::bfyx && in1_fmt == format::bfyx) {
                if (node.inputs_count() == 3) {
                    auto in2_dt = node.get_dependency(2).get_output_layout().data_type;
                    auto in2_fmt = node.get_dependency(2).get_output_layout().format;
                    if ((in2_dt == data_types::u8 || in2_dt == data_types::i8) &&
                        in2_fmt == format::bfyx)
                        does_support_fusings = true;
                    else
                        does_support_fusings = false;
                } else {
                    does_support_fusings = true;
                }
            }

            return does_support_fusings;
        };

        auto mvn_supports_fusings = [](mvn_node& node) -> bool {
            auto in_dt = node.get_dependency(0).get_output_layout().data_type;

            if (in_dt == data_types::u8 || in_dt == data_types::i8)
                return true;

            return false;
        };

        auto pooling_supports_fusings = [](pooling_node& node) -> bool {
            auto pooling_mode = node.as<pooling>().get_primitive()->mode;

            if (pooling_mode != cldnn::pooling_mode::max_with_argmax)
                return true;

            return false;
        };

        auto dts_supports_fusings = [](depth_to_space_node& node) -> bool {
            // Exclude `Conv -> DepthToSpace -> Eltwise (Sum)` case and handle it later by fusing into fused_conv_eltwise primitive
            bool input_conv = node.get_dependency(0).is_type<convolution>();
            bool out_eltw = node.get_users().front()->is_type<eltwise>();
            if (input_conv && out_eltw) {
                auto& eltw = static_cast<const eltwise&>(*node.get_users().front()->get_primitive());
                auto& conv = node.get_dependency(0).as<convolution>();
                auto eltw_mode = eltw.mode == eltwise_mode::sum;
                auto conv_size = conv.get_dependency(0).get_output_layout().size.spatial[0] % 128 == 0 &&
                                 conv.get_dependency(0).get_output_layout().size.spatial[1] % 2 == 0;
                auto format = conv.get_output_layout().format == format::bfyx;
                auto dt = conv.get_output_layout().data_type == data_types::f16;
                if (eltw_mode && conv_size && format && dt)
                    return false;
            }

            return true;
        };

        auto reduce_supports_fusings = [](reduce_node& node) -> bool {
            auto keep_dims = node.as<reduce>().get_primitive()->keep_dims;

            if (keep_dims)
                return true;

            return false;
        };

        auto eltwise_supports_fusings = [&](eltwise_node& node) -> bool {
            auto out_layout = node.get_output_layout();
            if (out_layout.data_type == data_types::f16 && out_layout.size.batch[0] > 1 &&
                (_lo.get_optimization_attributes().fs_b_yx_fsv32_network || out_layout.format == format::fs_b_yx_fsv32)) {
                return false;
            }

            return true;
        };

        auto get_users_from_fusing_history = [&](primitive_id id) {
            std::vector<primitive_id> users;
            for (auto deps_data : fusing_history) {
                auto key = deps_data.first;
                auto deps_vec = deps_data.second;
                auto iter = std::find(deps_vec.begin(), deps_vec.end(), id);
                if (iter != deps_vec.end()) {
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
                                return (desc.node->id() == prim_id);
                            });
                            if (origin_input_iter != fused_descs.end()) {
                                auto users = get_users_from_fusing_history(origin_input_iter->node->id());
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

            if (!input_data_supports_fusings(input_data, activation_node.id()))
                return;

            bool should_fuse = input_data.is_type<binary_convolution>();

            should_fuse |= input_data.is_type<convolution>() && conv_supports_fusings(input_data.as<convolution>());

            should_fuse |= input_data.is_type<fully_connected>() && fc_supports_fusings(input_data.as<fully_connected>());

            should_fuse |= input_data.is_type<gemm>() && gemm_supports_fusings(input_data.as<gemm>());

            should_fuse |= input_data.is_type<pooling>() && pooling_supports_fusings(input_data.as<pooling>());

            should_fuse |= input_data.is_type<resample>();

            should_fuse |= input_data.is_type<mvn>();

            should_fuse |= input_data.is_type<normalize>() &&
                          (input_data.get_dependency(0).get_output_layout().data_type == data_types::u8 ||
                           input_data.get_dependency(0).get_output_layout().data_type == data_types::i8);

            should_fuse |= input_data.is_type<deconvolution>();

            should_fuse |= input_data.is_type<permute>();

            should_fuse |= input_data.is_type<activation>();

            should_fuse |= input_data.is_type<lrn>();

            should_fuse |= input_data.is_type<gather>();

            should_fuse |= input_data.is_type<gather_nd>();

            should_fuse |= input_data.is_type<scatter_update>();

            should_fuse |= input_data.is_type<scatter_nd_update>();

            should_fuse |= input_data.is_type<scatter_elements_update>();

            should_fuse |= input_data.is_type<depth_to_space>();

            should_fuse |= input_data.is_type<space_to_depth>();

            should_fuse |= input_data.is_type<batch_to_space>();

            should_fuse |= input_data.is_type<space_to_batch>();

            should_fuse |= input_data.is_type<reduce>() && reduce_supports_fusings(input_data.as<reduce>());

            should_fuse |= input_data.is_type<scale>();

            // Here we need to check that Eltwise already has fused ops to avoid missing Activation primitive in
            // case `Conv -> Eltwise -> Activation` which will be replaced via fused_conv_eltwise primitive later
            // without handling any fused ops
            should_fuse |= input_data.is_type<eltwise>() && eltwise_supports_fusings(input_data.as<eltwise>()) && input_data.has_fused_primitives();

            if (!should_fuse)
                return;

            p.fuse_nodes(input_data, activation_node, &fusing_history);
        };

        auto fuse_scale_f = [&](scale_node& scale_node) {
            if (scale_node.get_dependencies().empty())
                CLDNN_ERROR_MESSAGE(scale_node.id(), "scale has invalid count of dependencies");

            auto& input_data = scale_node.get_dependency(0);
            if (input_data.get_users().size() != 1)
                return;

            bool should_fuse = input_data.is_type<binary_convolution>() &&
                               input_data.as<binary_convolution>().get_primitive()->dilation == tensor{1};

            should_fuse |= input_data.is_type<convolution>() && conv_supports_fusings(input_data.as<convolution>());

            should_fuse |= input_data.is_type<fully_connected>() && fc_supports_fusings(input_data.as<fully_connected>());

            should_fuse |= input_data.is_type<gemm>() && gemm_supports_fusings(input_data.as<gemm>());

            should_fuse |= input_data.is_type<pooling>() && pooling_supports_fusings(input_data.as<pooling>());

            should_fuse |= input_data.is_type<resample>();

            should_fuse |= input_data.is_type<mvn>() && mvn_supports_fusings(input_data.as<mvn>());

            should_fuse |= input_data.is_type<normalize>() &&
                          (input_data.get_dependency(0).get_output_layout().data_type == data_types::u8 ||
                           input_data.get_dependency(0).get_output_layout().data_type == data_types::i8);

            should_fuse |= input_data.is_type<deconvolution>();

            should_fuse |= input_data.is_type<permute>();

            should_fuse |= input_data.is_type<activation>();

            should_fuse |= input_data.is_type<lrn>();

            should_fuse |= input_data.is_type<gather>();

            should_fuse |= input_data.is_type<gather_nd>();

            should_fuse |= input_data.is_type<scatter_update>();

            should_fuse |= input_data.is_type<scatter_nd_update>();

            should_fuse |= input_data.is_type<scatter_elements_update>();

            should_fuse |= input_data.is_type<depth_to_space>();

            should_fuse |= input_data.is_type<space_to_depth>();

            should_fuse |= input_data.is_type<batch_to_space>();

            should_fuse |= input_data.is_type<space_to_batch>();

            should_fuse |= input_data.is_type<reduce>() && reduce_supports_fusings(input_data.as<reduce>());

            should_fuse |= input_data.is_type<scale>();

            should_fuse |= input_data.is_type<eltwise>() && eltwise_supports_fusings(input_data.as<eltwise>());

            if (!should_fuse)
                return;

            p.fuse_nodes(input_data, scale_node, &fusing_history);
        };

        auto fuse_quantize_f = [&](quantize_node& quantize_node) {
            auto& input_data = quantize_node.get_dependency(0);
            auto& input_lo = quantize_node.get_dependency(1);
            auto& input_hi = quantize_node.get_dependency(2);

            auto out_layout = quantize_node.get_output_layout();
            auto in_layout = input_data.get_output_layout();

            if (input_data.get_users().size() != 1)
                return;

            bool should_fuse = input_data.is_type<binary_convolution>() &&
                               ((out_layout.data_type == data_types::bin &&
                               quantize_node.get_dependencies().size() == 5 &&
                               ((in_layout.size.feature[0] == input_lo.get_output_layout().size.feature[0] &&
                                 in_layout.size.feature[0] == input_hi.get_output_layout().size.feature[0]) ||
                                (input_lo.get_output_layout().size.feature[0] == 1 &&
                                 input_hi.get_output_layout().size.feature[0] == 1)))) &&
                                 input_data.as<binary_convolution>().get_primitive()->dilation.spatial[0] == 1 &&
                                 input_data.as<binary_convolution>().get_primitive()->dilation.spatial[1] == 1;

            should_fuse |= input_data.is_type<convolution>() && conv_supports_fusings(input_data.as<convolution>()) &&
                           quantize_node.get_scale_shift_opt() &&
                           ((out_layout.data_type == data_types::f32 || out_layout.data_type == data_types::f16)  ||
                            input_data.get_output_layout().format == format::b_fs_yx_fsv16 ||
                            (_lo.should_select_b_fs_yx_fsv16_layout(input_data.as<convolution>(), input_data.get_dependency(1).get_output_layout()) &&
                             !is_grouped_conv(input_data.as<convolution>())) ||
                           // Avoid fusing to b_fs_yx_fsv16 (and similar) kernels
                           ((input_data.get_dependency(0).get_output_layout().data_type == data_types::u8 ||
                           input_data.get_dependency(0).get_output_layout().data_type == data_types::i8) &&
                           (out_layout.data_type == data_types::u8 || out_layout.data_type == data_types::i8)));

            should_fuse |= input_data.is_type<pooling>() && quantize_node.get_scale_shift_opt() &&
                           pooling_supports_fusings(input_data.as<pooling>());

            should_fuse |= input_data.is_type<fully_connected>() && fc_supports_fusings(input_data.as<fully_connected>()) &&
                           quantize_node.get_scale_shift_opt() &&
                           (out_layout.data_type == data_types::u8 || out_layout.data_type == data_types::i8);

            should_fuse |= input_data.is_type<lrn>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<gemm>() && gemm_supports_fusings(input_data.as<gemm>()) &&
                           quantize_node.get_scale_shift_opt() &&
                           (out_layout.data_type == data_types::u8 || out_layout.data_type == data_types::i8);

            should_fuse |= input_data.is_type<resample>() &&
                           quantize_node.get_scale_shift_opt() &&
                           (out_layout.data_type == data_types::u8 || out_layout.data_type == data_types::i8);

            should_fuse |= input_data.is_type<mvn>() && mvn_supports_fusings(input_data.as<mvn>()) &&
                           quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<activation>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<normalize>() && quantize_node.get_scale_shift_opt() &&
                          (input_data.get_dependency(0).get_output_layout().data_type == data_types::u8 ||
                           input_data.get_dependency(0).get_output_layout().data_type == data_types::i8);

            should_fuse |= input_data.is_type<deconvolution>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<gather>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<gather_nd>() && quantize_node.get_scale_shift_opt();

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

            should_fuse |= input_data.is_type<scale>() && quantize_node.get_scale_shift_opt();

            if (!should_fuse)
                return;

            p.fuse_nodes(input_data, quantize_node, &fusing_history);
        };

        auto fuse_eltwise_f = [&](eltwise_node& node) {
            std::shared_ptr<const cldnn::eltwise> prim = node.get_primitive();
            const std::vector<eltwise_mode> supported_modes = {
                eltwise_mode::sum,
                eltwise_mode::prod
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
                                      (parents[i]->is_type<scale>()) ||
                                      (parents[i]->is_type<gather_nd>()) ||
                                      (parents[i]->is_type<scatter_nd_update>()) ||
                                      (parents[i]->is_type<scatter_elements_update>()) ||
                                      (parents[i]->is_type<pooling>() && pooling_supports_fusings(parents[i]->as<pooling>())) ||
                                      (parents[i]->is_type<depth_to_space>() && dts_supports_fusings(parents[i]->as<depth_to_space>())) ||
                                      (parents[i]->is_type<reduce>() && reduce_supports_fusings(parents[i]->as<reduce>()));
            }

            // Disable fusion to a node on constant path when second input is in data flow
            for (size_t i = 0; i < parents.size(); i++) {
                can_fuse_parents[i] = can_fuse_parents[i] && (!parents[i]->is_constant() || parents[parents.size() - 1 - i]->is_constant());
            }

            auto parent1 = parents[0];
            auto parent2 = parents[1];

            auto p1_raw_size = parent1->get_output_layout().size.sizes();
            auto p2_raw_size = parent2->get_output_layout().size.sizes();
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
                                    !(user->is_type<activation>() && user->get_primitive()->input.size() == 1)));
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
            }

            for (auto& parent : fused_node->get_dependencies())
                if (parent->id() == peer_node->id())
                    merge_allowed = false;

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

        program_helpers::do_for_types<activation, scale, quantize, eltwise>(*node,
                fuse_activation_f,
                fuse_scale_f,
                fuse_quantize_f,
                fuse_eltwise_f);
    }

    // Need to update processing order to handle cases when peer node processing number is greater
    // than fused node one
    if (recalc_processing_order)
        p.get_processing_order().calc_processing_order(p);
}

void prepare_primitive_fusing::optimize_fused_ops(program_impl& p) {
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
                if (desc.node->id() == prim.node->id()) {
                    continue;
                }

                auto rm_iter = std::find_if(prim.fused_deps.begin(), prim.fused_deps.end(), [&](primitive_id& dep_id){
                    return (desc.node->id() == dep_id);
                });
                if (rm_iter != prim.fused_deps.end()) {
                    prim.fused_deps.erase(rm_iter);
                    prim.fused_deps.insert(prim.fused_deps.end(), desc.fused_deps.begin(), desc.fused_deps.end());
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

            if (fp.node->is_type<activation>() && fp_next.node->is_type<quantize>()) {
                auto& activation_node = fp.node->as<activation>();
                auto& quantize_node = fp_next.node->as<quantize>();
                bool can_skip = activation_node.get_primitive()->activation_function == activation_func::relu &&
                                activation_node.get_primitive()->additional_params.a == 0.0f &&
                                fp.deps.empty() &&
                                (quantize_node.get_output_layout().data_type == data_types::u8 ||
                                 quantize_node.get_output_layout().data_type == data_types::i8) &&
                                quantize_node.get_scale_shift_opt() &&
                                !quantize_node.get_need_pre_shift();

                if (can_skip) {
                    remove_deps_of_node(fp);
                    fp_itr = fused_prims.erase(curr_itr);
                }
            }
        }
    }
}

void prepare_conv_eltw_fusing::fuse_conv_depth_to_space(program_impl& p, program_node* node) {
    std::map<primitive_id, std::vector<primitive_id>> fusing_history;
    // make sure this convolution have only 1 user and it's depth_to_space
    // make sure convolution is not an output
    if (node->get_users().size() != 1 || node->is_output())
        return;

    if (!node->get_users().front()->is_type<depth_to_space>())
        return;

    convolution_node* conv_node = static_cast<convolution_node*>(node);

    depth_to_space_node* d_t_s_node = static_cast<depth_to_space_node*>(node->users.front());
    if (d_t_s_node->get_users().empty())
        return;
    if (!d_t_s_node->get_fused_primitives().empty())
        return;
    if (conv_node->get_dependency(0).get_output_layout().size.spatial[0] % 128 != 0 ||
        conv_node->get_dependency(0).get_output_layout().size.spatial[1] % 2 != 0)
        return;
    if (!d_t_s_node->get_users().front()->is_type<eltwise>())
        return;

    for (auto& dep : d_t_s_node->get_dependencies()) {
        format fmt = dep->get_output_layout().format;
        data_types dep_dt = dep->get_output_layout().data_type;
        if ((fmt != format::bfyx || dep_dt != data_types::f16))
            return;
    }

    p.fuse_nodes(*conv_node, *d_t_s_node, &fusing_history);
}

void prepare_conv_eltw_fusing::fuse_conv_eltwise(program_impl& p, program_node* node) {
    // make sure this convolution have only 1 user and it's eltwise
    // make sure convolution is not an output
    if (node->get_users().size() != 1 || node->is_output())
        return;

    if (!(*(node->get_users().begin()))->is_type<eltwise>())
        return;

    convolution_node* conv_node = static_cast<convolution_node*>(node);
    convolution& conv = const_cast<convolution&>(*conv_node->get_primitive());

    bool if_already_depth_to_space_fused = false;
    if (!conv_node->get_fused_primitives().empty())
        if_already_depth_to_space_fused = conv_node->get_fused_primitives().begin()->node->is_type<depth_to_space>();

    // TODO: find a better way to check for available kernels
    // currently works only for these formats
    data_types data_type = conv_node->get_output_layout().data_type;
    eltwise_node* eltw_node = static_cast<eltwise_node*>(*(node->users.begin()));

    if (eltw_node->has_fused_primitives())
        return;

    for (auto& dep : eltw_node->get_dependencies()) {
        format fmt = dep->get_output_layout().format;
        data_types dep_dt = dep->get_output_layout().data_type;
        if ((fmt != format::b_fs_yx_fsv4 || dep_dt != data_types::i8) &&
            (fmt != format::b_fs_yx_fsv4 || dep_dt != data_types::u8) &&
            (fmt != format::bfyx || dep_dt != data_types::f32) && (fmt != format::bfyx || dep_dt != data_types::u8) &&
            (fmt != format::bfyx || dep_dt != data_types::i8) && (fmt != format::yxfb || dep_dt != data_types::f16) &&
            (fmt != format::bfyx || dep_dt != data_types::f16 || !if_already_depth_to_space_fused))
            return;
    }

    auto weights_node_ptr = p.get_node_ptr(conv.weights[0]);
    auto filter_size = weights_node_ptr->get_output_layout().size;

    // Performance heuristic:
    // make sure that this is conv 1x1 with stride 1x1
    // disabled for i8 and u8 as those data_types currently must be fused
    if (data_type != data_types::u8 && data_type != data_types::i8) {
        if (filter_size.spatial[0] == 1 && filter_size.spatial[1] == 1) {
            if (conv.stride.spatial[0] != 1 || conv.stride.spatial[1] != 1)
                return;
        } else if (!if_already_depth_to_space_fused) {
            return;
        }
    }

    if (conv.groups != 1)
        return;

    // TODO Allow to pass arbitrary convolution activation in constructor
    if (!conv_node->get_fused_activations_funcs().empty() &&
        !(conv_node->get_fused_activations_funcs().size() == 1 && (conv_node->get_fused_activations_funcs()[0] == activation_func::relu ||
                                                                   conv_node->get_fused_activations_funcs()[0] == activation_func::relu_negative_slope ||
                                                                   conv_node->get_fused_activations_funcs()[0] == activation_func::none)))
        return;

    // make sure eltwise have only 2 inputs
    // make sure eltwise is not an output
    if (!if_already_depth_to_space_fused && (eltw_node->inputs_count() != 2 || eltw_node->is_output()))
        return;

    // only single ADD operation is currently supported
    // TODO: enable more
    eltwise& eltw = const_cast<eltwise&>(*eltw_node->get_primitive());
    if (eltw.mode != eltwise_mode::sum || !eltw.coefficients.empty())
        return;

    int eltw_fused_input_idx;   // <-- this input gets fused with eltwise
    int eltw_second_input_idx;  // <-- this input is not fused, so we add it in kernel

    if (eltw_node->input(0).is_type<convolution>()) {
        eltw_fused_input_idx = 0;
        eltw_second_input_idx = 1;
    } else {
        eltw_fused_input_idx = 1;
        eltw_second_input_idx = 0;
    }

    // we check if input to fuse is convolution that we're right now processing
    if (eltw_node->input(eltw_fused_input_idx).id() != conv.id)
        return;

    auto fused_output_layout_size = eltw_node->input(eltw_second_input_idx).get_output_layout().size;
    auto conv_output_layout_size = conv_node->get_output_layout().size;

    if (fused_output_layout_size.spatial[0] * fused_output_layout_size.spatial[1] * fused_output_layout_size.feature[0] * fused_output_layout_size.batch[0]
        != conv_output_layout_size.spatial[0] * conv_output_layout_size.spatial[1] * conv_output_layout_size.feature[0] * conv_output_layout_size.batch[0])
        return;

    // get strides for other than our conv input
    std::vector<tensor> new_eltw_strides;
    // conv strides modified by eltwise stride
    tensor new_conv_stride = conv.stride;

    if (eltw.stride.size() == eltw_node->inputs_count()) {
        // for cases when stride from eltwise must be applied into fused convolution
        new_conv_stride.spatial[0] *= eltw.stride[eltw_fused_input_idx].spatial[0];
        new_conv_stride.spatial[1] *= eltw.stride[eltw_fused_input_idx].spatial[1];
        // stride from non-fused eltwise input
        new_eltw_strides.push_back(eltw.stride[eltw_second_input_idx]);
    }

    auto conv_id = conv_node->id();
    auto eltw_id = eltw_node->id();

    bool conv_with_activation = !conv_node->get_fused_activations_funcs().empty();
    auto additional_params = conv_node->get_fused_activations_params();
    auto conv_netagive_slope = conv_with_activation && !additional_params.empty()
        ? additional_params.begin()->a : 0.0f;

    auto fused_conv_eltw =
        std::make_shared<fused_conv_eltwise>(conv_id + "_fused_" + eltw_id,
                                             conv_node->input().id(),
                                             eltw_node->input(eltw_second_input_idx).id(),
                                             eltw.mode,
                                             conv.weights,
                                             conv.bias,
                                             new_eltw_strides,
                                             new_conv_stride,
                                             conv.input_offset,
                                             conv.dilation,
                                             conv_with_activation,
                                             conv_netagive_slope,
                                             false,  // eltw.with_activation - use fused activation
                                             0.f);   // eltw.activation_negative_slope - use fused activation

    // Copy output data type from eltwise
    fused_conv_eltw->output_data_type = eltw_node->get_output_layout().data_type;

    fused_conv_eltw->depth_to_space_already_fused = if_already_depth_to_space_fused;

    auto& new_node = p.get_or_create(fused_conv_eltw);

    for (size_t i = 0; i < eltw_node->get_fused_activations_funcs().size(); i++)
        new_node.add_fused_activation(eltw_node->get_fused_activations_funcs()[i],
                                      eltw_node->get_fused_activations_params()[i]);

    p.replace(*eltw_node, new_node);

    // TODO: do it better, now it's done in a very ugly way to have good dependency order
    std::vector<program_node*> updated_deps;
    // Add convolution as dependency - will be replaced on extraction
    updated_deps.push_back(conv_node);

    // add second input
    updated_deps.push_back(&new_node.get_dependency(eltw_second_input_idx));

    // Copy convolution dependencies in order
    for (size_t d = 1; d < conv_node->get_dependencies().size(); d++) {
        updated_deps.push_back(&(conv_node->get_dependency(d)));
        conv_node->get_dependency(d).users.push_back(&new_node);
    }

    // Remove dependencies from convolution
    while (conv_node->get_dependencies().size() > 1) {
        conv_node->remove_dependency(1);
    }

    new_node.dependencies = updated_deps;

    if (if_already_depth_to_space_fused) {
        new_node.add_fused_primitives(conv_node->get_fused_primitives());
    }

    // Extract convolution node - will replace its usage in fused with input
    p.extract_and_remove(*conv_node);

    // To change convolution's output to image type, make sure that it is the last primitive in the topology,
    // or only reorder is afterwards and it is network's output
    auto reorder_user = (new_node.get_users().size() == 1);
    if (reorder_user)
        reorder_user &= ((new_node.get_users().front()->is_type<reorder>()) && (new_node.get_users().front()->is_output()));
    if (if_already_depth_to_space_fused && (new_node.get_users().size() == 0 || reorder_user)) {
        cldnn::layout new_layout = { data_types::u8, format::image_2d_rgba, fused_output_layout_size };
        new_node.set_output_layout(new_layout);
        // Remove output reorder if present
        if (reorder_user) {
            auto& reorder_node = new_node.get_users().front();
            reorder_node->remove_dependency(1);
            p.extract_and_remove(*reorder_node);
        }
    } else {
        new_node.recalc_output_layout();
    }

    p.add_optimized_primitive_info(conv_id, {new_node.id()});
    p.add_optimized_primitive_info(eltw_id, {new_node.id()});
}

void prepare_conv_eltw_fusing::run(program_impl& p) {
    std::list<program_node*> conv_nodes;
    // note we need to use iterators since currently processed element can be removed
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node_itr = itr++;
        if (node_itr != p.get_processing_order().end() &&
            (*node_itr)->is_type<convolution>())
            if (!b_fs_yx_fsv16_opt || !_lo.is_format_optimized((*node_itr)->as<convolution>(), format::b_fs_yx_fsv16))
                conv_nodes.push_back(*node_itr);
    }

    // fuse conv + eltwise after activations
    auto conv_itr = conv_nodes.begin();
    while (conv_itr != conv_nodes.end()) {
        auto node_itr = conv_itr++;

        if (node_itr == conv_nodes.end())
            break;

        auto& node = (*node_itr);

        fuse_conv_depth_to_space(p, node);

        fuse_conv_eltwise(p, node);
    }
}

void prepare_conv_eltw_read_write_opt::conv_eltwise_read_write_opt(program_impl& p, program_node* node) {
    fused_conv_eltwise_node* fused_conv_eltw_node = static_cast<fused_conv_eltwise_node*>(node);
    program_node* second_input_node = &fused_conv_eltw_node->get_dependency(1);
    // output layouts must match
    if (fused_conv_eltw_node->get_output_layout() != second_input_node->get_output_layout()) {  // check whole layout
        return;
    }

    // look for conflicts
    auto this_node_processing_number = p.get_processing_order().get_processing_number(node);
    for (auto& user : second_input_node->users) {
        if (p.get_processing_order().get_processing_number(user) > this_node_processing_number)
            return;
    }

    // buffer shared between primitives, if second input is mutable data, then we can reuse this memory
    auto shared_buffer_mem = second_input_node->is_type<mutable_data>()
                                 ? second_input_node->as<mutable_data>().get_attached_memory_ptr()
                                 : p.get_engine().allocate_memory(node->get_output_layout(), 0);

    float zero = 0.0f;
    layout dummy_layout(data_types::f32, format::bfyx, tensor(1, 1, 1, 1));

    // this one is the first one to write data to
    auto rw_output_prim0 = std::make_shared<mutable_data>(fused_conv_eltw_node->id() + "_RW_OPT_use",
                                                          memory::attach(dummy_layout, &zero, 1));
    // this one already expects data to be inside
    auto rw_output_prim1 = std::make_shared<mutable_data>(fused_conv_eltw_node->id() + "_RW_OPT_reuse",
                                                          memory::attach(dummy_layout, &zero, 1));

    auto& rw_output_node0 = p.get_or_create(rw_output_prim0);
    auto& rw_output_node1 = p.get_or_create(rw_output_prim1);

    rw_output_node0.as<mutable_data>().attach_memory(*shared_buffer_mem, false);
    rw_output_node1.as<mutable_data>().attach_memory(*shared_buffer_mem, false);

    // add connection between second input node -> rw_output_node0 -> node
    p.add_intermediate(rw_output_node0, *node, 1, true);
    // replace other connections with rw_output_node0
    auto itr = second_input_node->users.begin();
    while (itr != second_input_node->users.end()) {
        auto& usage = (*itr++);
        if (usage->id() != rw_output_node0.id() && usage->id() != node->id()) {
            usage->replace_dependency(*second_input_node, rw_output_node0);
        }
    }
    // add connection between node -> rw_output_node1 -> after nodes
    // first find index in our first user's dependency
    size_t dep_idx = 0;
    for (auto dep : (*(node->users.begin()))->dependencies) {
        if (dep->id() == node->id())
            break;
        dep_idx++;
    }
    p.add_intermediate(rw_output_node1, **(node->users.begin()), dep_idx, true);
    // replace other connections with rw_output_node1
    itr = node->users.begin();
    while (itr != node->users.end()) {
        auto& usage = (*itr++);
        if (usage->id() != rw_output_node1.id() && usage->id() != node->id()) {
            usage->replace_dependency(*node, rw_output_node1);
        }
    }
    fused_conv_eltwise* prim = const_cast<fused_conv_eltwise*>((fused_conv_eltw_node->get_primitive().get()));
    prim->second_input_in_output = true;
}

void prepare_conv_eltw_read_write_opt::run(program_impl& p) {
    std::list<program_node*> fused_conv_eltw_nodes;
    auto itr = p.get_processing_order()
                   .begin();  // note we need to use iterators since currently processed element can be removed
    while (itr != p.get_processing_order().end()) {
        auto node_itr = itr++;
        if (node_itr != p.get_processing_order().end() &&
            (*node_itr)->is_type<fused_conv_eltwise>())
            fused_conv_eltw_nodes.push_back(*node_itr);
    }

    // fuse conv + eltwise after activations
    itr = fused_conv_eltw_nodes.begin();
    while (itr != fused_conv_eltw_nodes.end()) {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        conv_eltwise_read_write_opt(p, node);
    }
}
