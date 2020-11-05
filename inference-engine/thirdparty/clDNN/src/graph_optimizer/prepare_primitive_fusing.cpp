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
#include "scale_inst.h"
#include "resample_inst.h"
#include "depth_to_space_inst.h"
#include "space_to_depth_inst.h"
#include "gather_inst.h"
#include "scatter_update_inst.h"
#include "reverse_sequence_inst.h"
#include "shuffle_channels_inst.h"
#include "space_to_batch_inst.h"
#include "strided_slice_inst.h"
#include "cum_sum_inst.h"
#include "embedding_bag_inst.h"
#include "extract_image_patches_inst.h"
#include "reduce_inst.h"
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
                (!input.is_type<concatenation>() && !input.is_type<convolution>() &&
                 !input.is_type<crop>() && !input.is_type<deconvolution>() && !input.is_type<eltwise>() &&
                 !input.is_type<fully_connected>() && !input.is_type<lrn>() && !input.is_type<normalize>() &&
                 !input.is_type<permute>() && !input.is_type<pooling>() && !input.is_type<reorder>() &&
                 !input.is_type<reshape>() && !input.is_type<roi_pooling>() && !input.is_type<scale>() &&
                 !input.is_type<softmax>() && !input.is_type<resample>() && !input.is_type<mvn>() &&
                 !input.is_type<depth_to_space>() && !input.is_type<batch_to_space>() &&
                 !input.is_type<space_to_batch>() && !input.is_type<gather>() && !input.is_type<scatter_update>() && !input.is_type<shuffle_channels>() &&
                 !input.is_type<strided_slice>() && !input.is_type<cum_sum>() && !input.is_type<reverse_sequence>() &&
                 !input.is_type<embedding_bag>() && !input.is_type<extract_image_patches>() && !input.is_type<activation>()))
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

            should_fuse |= input_data.is_type<scatter_update>();

            should_fuse |= input_data.is_type<depth_to_space>();

            should_fuse |= input_data.is_type<space_to_depth>();

            should_fuse |= input_data.is_type<batch_to_space>();

            should_fuse |= input_data.is_type<space_to_batch>();

            should_fuse |= input_data.is_type<reduce>() && reduce_supports_fusings(input_data.as<reduce>());

            should_fuse |= input_data.is_type<scale>();

            should_fuse |= input_data.is_type<eltwise>() && eltwise_supports_fusings(input_data.as<eltwise>());

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

            should_fuse |= input_data.is_type<scatter_update>();

            should_fuse |= input_data.is_type<depth_to_space>();

            should_fuse |= input_data.is_type<space_to_depth>();

            should_fuse |= input_data.is_type<batch_to_space>();

            should_fuse |= input_data.is_type<space_to_batch>();

            should_fuse |= input_data.is_type<reduce>() && reduce_supports_fusings(input_data.as<reduce>());

            should_fuse |= input_data.is_type<scale>();

            should_fuse |= input_data.is_type<eltwise>() && eltwise_supports_fusings(input_data.as<eltwise>());

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

            should_fuse |= input_data.is_type<scatter_update>() && quantize_node.get_scale_shift_opt();

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

            p.fuse_nodes(input_data, quantize_node);
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
                                      (parents[i]->is_type<mvn>() && mvn_supports_fusings(parents[i]->as<mvn>())) ||
                                      (parents[i]->is_type<deconvolution>()) ||
                                      (parents[i]->is_type<permute>()) ||
                                      (parents[i]->is_type<space_to_depth>()) ||
                                      (parents[i]->is_type<gemm>() && gemm_supports_fusings(parents[i]->as<gemm>())) ||
                                      (parents[i]->is_type<batch_to_space>()) ||
                                      (parents[i]->is_type<space_to_batch>()) ||
                                      (parents[i]->is_type<eltwise>() && eltwise_supports_fusings(parents[i]->as<eltwise>())) ||
                                      (parents[i]->is_type<scale>()) ||
                                      (parents[i]->is_type<depth_to_space>() && dts_supports_fusings(parents[i]->as<depth_to_space>())) ||
                                      (parents[i]->is_type<reduce>() && reduce_supports_fusings(parents[i]->as<reduce>()));
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
                }
                else if (p2_raw_size[k] < p1_raw_size[k]) {
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

            // This fusing can be extended to support peer node in any layout
            bool merge_allowed = fused_node->get_users().size() == 1;

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
                    fp_itr = fused_prims.erase(curr_itr);
                }
            }
        }
    }
}
