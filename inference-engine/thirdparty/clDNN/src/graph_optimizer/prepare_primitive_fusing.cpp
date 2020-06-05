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

#include "api/pooling.hpp"
#include "api/proposal.hpp"
#include "api/roi_pooling.hpp"

#include "program_helpers.h"
#include "pass_manager.h"

#include "quantize_inst.h"
#include "binary_convolution_inst.h"
#include "activation_inst.h"
#include "batch_norm_inst.h"
#include "batch_norm_grad_inst.h"
#include "crop_inst.h"
#include "eltwise_inst.h"
#include "fused_conv_bn_scale_inst.h"
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
#include "scale_grad_weights_inst.h"
#include "resample_inst.h"
#include "depth_to_space_inst.h"
#include "gather_inst.h"
#include "reverse_sequence_inst.h"
#include "shuffle_channels_inst.h"
#include "strided_slice_inst.h"
#include "cum_sum_inst.h"
#include "embedding_bag_inst.h"
#include <vector>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include "error_handler.h"

void prepare_primitive_fusing::run(program_impl& p) {
    fuse_reorders(p);
    fuse_sigmoid_mul_to_swish(p);
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

            auto swish_prim = std::make_shared<cldnn::activation>(mul.id()+"_swish", input.id(), activation_func::swish);
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
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node_itr = itr++;
        auto& node = (*node_itr);

        program_helpers::do_for_types<activation>(*node, [&p, &is_debug](activation_node& node) {
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
                (!input.is_type<batch_norm>() && !input.is_type<concatenation>() && !input.is_type<convolution>() &&
                 !input.is_type<crop>() && !input.is_type<deconvolution>() && !input.is_type<eltwise>() &&
                 !input.is_type<fully_connected>() && !input.is_type<lrn>() && !input.is_type<normalize>() &&
                 !input.is_type<permute>() && !input.is_type<pooling>() && !input.is_type<reorder>() &&
                 !input.is_type<reshape>() && !input.is_type<roi_pooling>() && !input.is_type<scale>() &&
                 !input.is_type<softmax>() && !input.is_type<resample>() && !input.is_type<mvn>() &&
                 !input.is_type<depth_to_space>() && !input.is_type<gather>() && !input.is_type<reverse_sequence>() &&
                 !input.is_type<shuffle_channels>() && !input.is_type<strided_slice>() && !input.is_type<cum_sum>() &&
                 !input.is_type<embedding_bag>() && !input.is_type<fused_conv_eltwise>() &&
                 !input.is_type<activation>()))
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
                p.fuse_nodes(input, node);
            }

            p.add_optimized_primitive_info(id, {input.id()});
        });
    }
}

void prepare_primitive_fusing::fuse_simple_primitives(program_impl &p) {
    bool recalc_processing_order = false;

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
            if ((in0_dt == data_types::u8 || in0_dt == data_types::i8) &&
                (in1_dt == data_types::u8 || in1_dt == data_types::i8) &&
                in0_fmt == format::bfyx && in1_fmt == format::bfyx)
                does_support_fusings = true;

            if (node.inputs_count() == 3) {
                auto in2_dt = node.get_dependency(2).get_output_layout().data_type;
                auto in2_fmt = node.get_dependency(2).get_output_layout().format;
                if ((in2_dt == data_types::u8 || in2_dt == data_types::i8) &&
                    in2_fmt == format::bfyx)
                    does_support_fusings = true;
                else
                    does_support_fusings = false;
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

        auto fuse_activation_f = [&](activation_node& activation_node) {
            auto& input_data = activation_node.get_dependency(0);
            if (input_data.get_users().size() != 1 || activation_node.get_dependencies().size() >= 3)
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

            if (!should_fuse)
                return;

            p.fuse_nodes(input_data, activation_node);
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

            if (!should_fuse)
                return;

            p.fuse_nodes(input_data, scale_node);
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

            should_fuse |= input_data.is_type<deconvolution>() && quantize_node.get_scale_shift_opt() &&
                            // fp16/fp32 optimized kernels don't support chaning data type
                           (input_data.get_dependency(0).get_output_layout().data_type == data_types::u8 ||
                            input_data.get_dependency(0).get_output_layout().data_type == data_types::i8 ||
                            input_data.get_output_layout().data_type == out_layout.data_type);
            
            should_fuse |= input_data.is_type<gather>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<permute>() &&
                           quantize_node.get_scale_shift_opt();

            if (!should_fuse)
                return;

            p.fuse_nodes(input_data, quantize_node);
        };

        auto fuse_eltwise_f = [&](eltwise_node& node) {
            std::shared_ptr<const cldnn::eltwise> prim = node.get_primitive();
            if (node.is_output() || node.inputs_count() != 2 ||
                prim->mode != eltwise_mode::sum || !prim->stride.empty())
                return;

            std::vector<cldnn::program_node*> parents = node.get_dependencies();
            std::list<cldnn::program_node*> users = node.get_users();

            auto parent1 = parents[0];
            auto parent2 = parents[1];

            bool can_fuse_parent1 = (parent1->is_type<convolution>() && conv_supports_fusings(parent1->as<convolution>())) ||
                                    (parent1->is_type<mvn>() && mvn_supports_fusings(parent1->as<mvn>())) ||
                                    (parent1->is_type<deconvolution>()) || (parent1->is_type<permute>());

            bool can_fuse_parent2 = (parent2->is_type<convolution>() && conv_supports_fusings(parent2->as<convolution>())) ||
                                    (parent2->is_type<mvn>() && mvn_supports_fusings(parent2->as<mvn>())) ||
                                    (parent2->is_type<deconvolution>()) || (parent2->is_type<permute>());

            std::vector<bool> can_fuse_parents = { can_fuse_parent1, can_fuse_parent2 };

            // We should have at least one node to fuse
            if (!can_fuse_parent1 && !can_fuse_parent2)
                return;

            // Choose node to fuse
            size_t fused_idx = can_fuse_parent1 ? 0 : 1;
            size_t peer_idx  = can_fuse_parent1 ? 1 : 0;

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

            // This fusing can be extended to support peer node in any layout and with broadcast
            bool merge_allowed = fused_node->get_users().size() == 1 &&
                                 fused_node->get_output_layout().size == peer_node->get_output_layout().size;

            for (auto& parent : fused_node->get_dependencies())
                if (parent->id() == peer_node->id())
                    merge_allowed = false;

            if (!merge_allowed)
                return;

            if (p.get_processing_order().get_processing_number(fused_node) <
                p.get_processing_order().get_processing_number(peer_node))
                recalc_processing_order = true;

            p.fuse_nodes(*fused_node, node);
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
                    fused_prims.erase(curr_itr);
                }
            }
        }
    }
}

void prepare_conv_eltw_fusing::fuse_conv_depth_to_space(program_impl& p, program_node* node) {
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
    if (!d_t_s_node->get_users().front()->is_type<eltwise>())
        return;

    for (auto& dep : d_t_s_node->get_dependencies()) {
        format fmt = dep->get_output_layout().format;
        data_types dep_dt = dep->get_output_layout().data_type;
        if ((fmt != format::bfyx || dep_dt != data_types::f16))
            return;
    }

    p.fuse_nodes(*conv_node, *d_t_s_node);
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
    for (auto& dep : eltw_node->get_dependencies()) {
        format fmt = dep->get_output_layout().format;
        data_types dep_dt = dep->get_output_layout().data_type;
        if ((fmt != format::fs_bs_yx_bsv4_fsv32 || dep_dt != data_types::i8) &&
            (fmt != format::b_fs_yx_fsv4 || dep_dt != data_types::i8) &&
            (fmt != format::b_fs_yx_fsv4 || dep_dt != data_types::u8) &&
            (fmt != format::byxf_af32 || dep_dt != data_types::i8) &&
            (fmt != format::byxf_af32 || dep_dt != data_types::u8) &&
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

    // Get scaling of second eltwise input - only per tensor supported for now
    float eltw_scale = 1.f;

    if (eltw_node->inputs_quantization_term()) {
        eltw_scale = eltw.input_quantization_factors[eltw_second_input_idx] /
                     eltw.input_quantization_factors[eltw_fused_input_idx];
    }

    if (eltw_node->inputs_calibration_term())
        return;

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
                                             std::vector<primitive_id>{},
                                             std::vector<primitive_id>{},
                                             0.0f,
                                             eltw_scale,  // eltw_scale
                                             eltw.output_calibration_factors,
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

    // Copy output calibration factors pointer as replace will remove eltwise node
    program_node* output_calibration_factors = nullptr;
    if (eltw_node->output_calibration_term()) {
        output_calibration_factors = &eltw_node->output_calibration_factors();
    }

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

    if (output_calibration_factors != nullptr) {
        updated_deps.push_back(output_calibration_factors);
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
