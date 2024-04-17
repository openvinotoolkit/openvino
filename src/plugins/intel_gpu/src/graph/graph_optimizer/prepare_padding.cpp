// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_inst.h"
#include "program_node.h"
#include "pass_manager.h"
#include "convolution_inst.h"
#include "mvn_inst.h"
#include "sliding_window_utils.hpp"
#include <algorithm>

using namespace cldnn;
using namespace ov::intel_gpu;

void prepare_padding::run(program& p) {
    if (output_size_handling_enabled) {
        // Prepare upper padding for primitives that support output_size parameter.
        for (const auto& node : p.get_processing_order()) {
            if (node->get_dependencies().empty())
                continue;

            if (node->get_dependency(0).is_type<data>())
                continue;

            // Padded offsets aren't supported by onednn kernels
            if (node->get_preferred_impl_type() == impl_types::onednn)
                continue;

            auto add_required_padding = [&p](program_node& node, padding& needed_padding) {
                // Add extra reorder for cldnn primitive to handle required padding if needed
                auto& input = node.get_dependency(0);
                bool is_usr_onednn = false;
                for (auto& input_usr : input.get_users())
                    if (input_usr->get_preferred_impl_type() == impl_types::onednn)
                        is_usr_onednn = true;

                if ((input.get_preferred_impl_type() == impl_types::onednn || is_usr_onednn) &&
                    node.get_preferred_impl_type() == impl_types::ocl &&
                    static_cast<bool>(needed_padding)) {
                    auto new_reorder = std::make_shared<reorder>(node.id() + "_padding_reorder_for_" + input.id(), input.id(), input.get_output_layout());
                    auto& new_reorder_node = p.get_or_create(new_reorder);
                    p.add_intermediate(new_reorder_node, node, input);
                }

                p.apply_needed_padding(node, node.get_dependency(0), needed_padding);
            };

            if (node->is_type<convolution>()) {
                auto& prim_node = node->as<convolution>();
                const auto& prim = prim_node.get_primitive();

                auto format = node->get_output_layout().format;
                if (format == format::b_fs_zyx_fsv16 ||
                    format == format::bs_fs_zyx_bsv16_fsv16 ||
                    format == format::bs_fs_yx_bsv16_fsv16 ||
                    format == format::bs_fs_yx_bsv32_fsv32 ||
                    format == format::b_fs_zyx_fsv32)
                    continue;

                auto padding_begin = prim->padding_begin;
                auto padding_end = prim->padding_end;

                tensor::value_type pb_z = std::max<std::ptrdiff_t>(padding_begin.size() >= 3 ? padding_begin[padding_begin.size() - 3] : 0, 0);
                tensor::value_type pb_y = std::max<std::ptrdiff_t>(padding_begin.size() >= 2 ? padding_begin[padding_begin.size() - 2] : 0, 0);
                tensor::value_type pb_x = std::max<std::ptrdiff_t>(padding_begin.size() >= 1 ? padding_begin[padding_begin.size() - 1] : 0, 0);

                tensor::value_type pe_z = std::max<std::ptrdiff_t>(padding_end.size() >= 3 ? padding_end[padding_end.size() - 3] : 0, 0);
                tensor::value_type pe_y = std::max<std::ptrdiff_t>(padding_end.size() >= 2 ? padding_end[padding_end.size() - 2] : 0, 0);
                tensor::value_type pe_x = std::max<std::ptrdiff_t>(padding_end.size() >= 1 ? padding_end[padding_end.size() - 1] : 0, 0);

                tensor pad_l = tensor(0);
                tensor pad_u = tensor(0);
                pad_l.spatial[0] = pb_x;
                pad_l.spatial[1] = pb_y;
                pad_l.spatial[2] = pb_z;

                pad_u.spatial[0] = pe_x;
                pad_u.spatial[1] = pe_y;
                pad_u.spatial[2] = pe_z;

                auto in_layout = prim_node.get_input_layout();

                const auto& actual_lpad = in_layout.data_padding.lower_size();
                const auto& actual_upad = in_layout.data_padding.upper_size();

                auto needed_lpad = tensor::max(pad_l, actual_lpad);
                auto needed_upad = tensor::max(pad_u, actual_upad);

                padding needed_padding(needed_lpad.sizes(), needed_upad.sizes());

                add_required_padding(prim_node, needed_padding);
            } else if (node->is_type<deconvolution>()) {
                auto& prim_node = node->as<deconvolution>();
                const auto& prim = prim_node.get_primitive();

                if (!prim->with_output_size)
                    continue;

                auto filter_size = prim_node.weights().get_output_layout().get_tensor();

                auto needed_padding = calc_sliding_window_needed_input_padding(prim_node.get_input_layout(),
                                                                               prim->output_size,
                                                                               filter_size,
                                                                               prim->pad,
                                                                               prim->stride,
                                                                               ov::Strides(prim->stride.size(), 1),
                                                                               true,
                                                                               1);

                add_required_padding(prim_node, needed_padding);
            } else if (node->is_type<pooling>()) {
                auto& prim_node = node->as<pooling>();
                const auto& prim = prim_node.get_primitive();

                if (!prim->with_output_size)
                    continue;

                padding needed_padding;
                // WA for this format. sliding window needs to be fixed --perf degradation for IncepctionV1 type models
                tensor size(1);
                for (size_t i = 0; i < prim->size.size(); i++) {
                    size.spatial[i] = static_cast<tensor::value_type>(prim->size[prim->size.size() - i - 1]);
                }

                if (node->get_output_layout().format == format::b_fs_yx_fsv16)
                    needed_padding = calc_sliding_window_needed_input_padding(prim_node.get_input_layout(),
                                                                              prim->output_size,
                                                                              size,
                                                                              ov::CoordinateDiff(prim->pads_begin.begin(), prim->pads_begin.end()),
                                                                              prim->stride,
                                                                              ov::Strides(prim->size.size(), 1),
                                                                              false,
                                                                              1);
                else
                    needed_padding = prim_node.get_input_layout().data_padding;

                add_required_padding(prim_node, needed_padding);
            }
        }
    }

    // Prepare optimized padding for bfyx convolution.
    for (auto& pair : p.nodes_map) {
        if (pair.second->type() != convolution::type_id())
            continue;

        auto& node = pair.second->as<convolution>();
        if (node.get_dependencies().empty())
            continue;

        auto conv = node.get_primitive();

        if (node.is_dynamic() && !node.use_explicit_padding())
            continue;

        auto& conv_input_node = node.get_dependency(0);
        auto conv_layout = node.get_output_layout();

        // right now output padding optimization is only available for bfyx format and data type = float32
        if (conv_layout.format != cldnn::format::bfyx &&
            conv_layout.format != cldnn::format::b_fs_yx_fsv16 &&
            conv_layout.format != cldnn::format::b_fs_zyx_fsv16 &&
            conv_layout.format != cldnn::format::bs_fs_yx_bsv16_fsv16 &&
            conv_layout.format != cldnn::format::b_fs_yx_fsv4 &&
            conv_layout.format != cldnn::format::fs_b_yx_fsv32) {
            continue;
        }

        // convolution have only one input primitive
        auto prev_prim_output_layout = conv_input_node.get_output_layout();

        // For 3d convolution padding is needed only for int8 case
        // FP16/32 kernels can work w/o physical padding
        if (prev_prim_output_layout.format == cldnn::format::b_fs_zyx_fsv16 &&
            prev_prim_output_layout.data_type != data_types::i8 && prev_prim_output_layout.data_type != data_types::u8)
            continue;

        // We shoudn't apply any padding to nodes which are marked as outputs or have type as data
        if (conv_input_node.is_output() || conv_input_node.is_type<data>())
            continue;

        // Padded offsets aren't supported by onednn kernels
        if (conv_input_node.get_preferred_impl_type() == impl_types::onednn)
            continue;

        if (node.get_preferred_impl_type() == impl_types::onednn)
            continue;

        // Calculating input padding needed for convolution
        auto& filter_node = node.as<convolution>().weights();
        auto filter_prim = filter_node.get_primitive();

        layout filter_layout = filter_node.get_output_layout().convert_to_weights_layout(conv->grouped_weights_shape);

        // Compute initial required paddings for primitive used as input for convolution.
        auto padding_begin = conv->padding_begin;
        auto padding_end = conv->padding_end;
        auto stride = conv->stride;
        auto dilation = conv->dilation;
        uint32_t stride_z = stride.size() >= 3 ? static_cast<uint32_t>(stride[stride.size() - 3]) : 1;
        uint32_t stride_y = stride.size() >= 2 ? static_cast<uint32_t>(stride[stride.size() - 2]) : 1;
        uint32_t stride_x = stride.size() >= 1 ? static_cast<uint32_t>(stride[stride.size() - 1]) : 1;

        uint32_t dilation_z = dilation.size() >= 3 ? static_cast<uint32_t>(dilation[dilation.size() - 3]) : 1;
        uint32_t dilation_y = dilation.size() >= 2 ? static_cast<uint32_t>(dilation[dilation.size() - 2]) : 1;
        uint32_t dilation_x = dilation.size() >= 1 ? static_cast<uint32_t>(dilation[dilation.size() - 1]) : 1;

        tensor::value_type pad_z = padding_begin.size() >= 3 ? padding_begin[padding_begin.size() - 3] : 0;
        tensor::value_type pad_y = padding_begin.size() >= 2 ? padding_begin[padding_begin.size() - 2] : 0;
        tensor::value_type pad_x = padding_begin.size() >= 1 ? padding_begin[padding_begin.size() - 1] : 0;

        tensor::value_type padding_begin_x, padding_begin_y, padding_begin_z;
        tensor::value_type padding_end_x, padding_end_y, padding_end_z;

        if (node.is_dynamic() && node.use_explicit_padding()) {
            padding_begin_x = std::max(pad_x, 0);
            padding_begin_y = std::max(pad_y, 0);
            padding_begin_z = std::max(pad_z, 0);

            pad_z = padding_end.size() >= 3 ? padding_end[padding_end.size() - 3] : 0;
            pad_y = padding_end.size() >= 2 ? padding_end[padding_end.size() - 2] : 0;
            pad_x = padding_end.size() >= 1 ? padding_end[padding_end.size() - 1] : 0;

            padding_end_x = std::max(pad_x, 0);
            padding_end_y = std::max(pad_y, 0);
            padding_end_z = std::max(pad_z, 0);
        } else {
            auto input_limit_x = -pad_x + (conv_layout.spatial(0) - 1) * stride_x +
                                (filter_layout.spatial(0) - 1) * dilation_x + 1;
            auto input_limit_y = -pad_y + (conv_layout.spatial(1) - 1) * stride_y +
                                (filter_layout.spatial(1) - 1) * dilation_y + 1;
            auto input_limit_z = -pad_z + (conv_layout.spatial(2) - 1) * stride_z +
                                (filter_layout.spatial(2) - 1) * dilation_z + 1;

            padding_begin_x = std::max(pad_x, 0);
            padding_begin_y = std::max(pad_y, 0);
            padding_begin_z = std::max(pad_z, 0);
            padding_end_x = std::max<tensor::value_type>(input_limit_x - prev_prim_output_layout.spatial(0), 0);
            padding_end_y = std::max<tensor::value_type>(input_limit_y - prev_prim_output_layout.spatial(1), 0);
            padding_end_z = std::max<tensor::value_type>(input_limit_z - prev_prim_output_layout.spatial(2), 0);
        }

        auto& input = node.get_dependency(0);
        // WA to add reorder between MVN and Conv because Conv need input data with padding but MVN opt kernel with default format does not support padding.
        // TODO: MVN opt kernel should support padding.
        if (node.get_preferred_impl_type() == impl_types::ocl && input.is_type<mvn>()
            && format::is_default_format(input.get_output_layout().format)) { // check the allowed format to avoid perf drop by unnecessary reorder addition.
            auto new_reorder = std::make_shared<reorder>(node.id() + "_padding_reorder_for_" + input.id(), input.id(), input.get_output_layout());
            auto& new_reorder_node = p.get_or_create(new_reorder);
            p.add_intermediate(new_reorder_node, node, input);
        }

        // Adjust right padding, so entire buffer size in X dimension is properly aligned.
        // TODO: NOTE: Will be reenabled with next check-in once heuristic for line-aligned algorithm will be added.
        // auto needed_buffer_size_x = static_cast<cldnn::tensor::value_type>(
        //    round_up_to(left_padding + prev_prim_output_layout.spatial(0) + right_padding, 16));
        // right_padding = needed_buffer_size_x - left_padding - prev_prim_output_layout.spatial(0);

        cldnn::padding needed_padding({0, 0, padding_begin_x, padding_begin_y, padding_begin_z}, {0, 0, padding_end_x, padding_end_y, padding_end_z}, 0);
        needed_padding = padding::max(prev_prim_output_layout.data_padding, needed_padding);
        p.apply_needed_padding(node, node.get_dependency(0), needed_padding);
    }
}
