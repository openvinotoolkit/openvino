// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pooling_inst.h"
#include "program_node.h"
#include "pass_manager.h"
#include "convolution_inst.h"
#include "sliding_window_utils.h"
#include <algorithm>

using namespace cldnn;

void prepare_padding::run(program_impl& p) {
    if (output_size_handling_enabled) {
        // Prepare upper padding for primitives that support output_size parameter.
        for (const auto& node : p.get_processing_order()) {
            if (node->is_type<convolution>()) {
                auto& prim_node = node->as<convolution>();
                const auto& prim = prim_node.get_primitive();

                if (!prim->with_output_size)
                    continue;

                auto format = node->get_output_layout().format;
                if (format == format::b_fs_zyx_fsv16 ||
                    format == format::bs_fs_zyx_bsv16_fsv16 ||
                    format == format::bs_fs_yx_bsv16_fsv16 ||
                    format == format::b_fs_zyx_fsv32)
                    continue;

                if (prim_node.input().is_type<data>()) {
                    continue;
                }

                auto filter_size = prim_node.weights(0).get_output_layout().size;

                auto needed_padding = calc_sliding_window_needed_input_padding(prim_node.input().get_output_layout(),
                                                                               prim->output_size,
                                                                               filter_size,
                                                                               prim->input_offset,
                                                                               prim->stride,
                                                                               prim->dilation,
                                                                               false,
                                                                               1);

                p.apply_needed_padding(prim_node, prim_node.input(), needed_padding);
            } else if (node->is_type<deconvolution>()) {
                auto& prim_node = node->as<deconvolution>();
                const auto& prim = prim_node.get_primitive();

                if (!prim->with_output_size)
                    continue;

                if (prim_node.input().is_type<data>()) {
                    continue;
                }

                auto filter_size = prim_node.weights(0).get_output_layout().size;

                auto needed_padding = calc_sliding_window_needed_input_padding(prim_node.input().get_output_layout(),
                                                                               prim->output_size,
                                                                               filter_size,
                                                                               prim->input_offset,
                                                                               prim->stride,
                                                                               {1, 1, 1, 1},
                                                                               true,
                                                                               1);

                p.apply_needed_padding(prim_node, prim_node.input(), needed_padding);
            } else if (node->is_type<pooling>()) {
                auto& prim_node = node->as<pooling>();
                const auto& prim = prim_node.get_primitive();

                if (!prim->with_output_size)
                    continue;

                if (prim_node.input().is_type<data>()) {
                    continue;
                }

                padding needed_padding;
                // WA for this format. sliding window needs to be fixed --perf degradation for IncepctionV1 type models
                if (node->get_output_layout().format == format::b_fs_yx_fsv16)
                    needed_padding = calc_sliding_window_needed_input_padding(prim_node.input().get_output_layout(),
                                                                              prim->output_size,
                                                                              prim->size,
                                                                              prim->input_offset,
                                                                              prim->stride,
                                                                              {1, 1, 1, 1},
                                                                              false,
                                                                              1);
                else
                    needed_padding = prim_node.input().get_output_layout().data_padding;

                p.apply_needed_padding(prim_node, prim_node.input(), needed_padding);
            } else if (node->is_type<binary_convolution>()) {
                auto& prim_node = node->as<binary_convolution>();

                if (prim_node.input().is_type<data>()) {
                    continue;
                }

                auto needed_padding = prim_node.input().get_output_layout().data_padding;

                p.apply_needed_padding(prim_node, prim_node.input(), needed_padding);
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
        auto& conv_input_node = node.get_dependency(0);
        auto conv_layout = node.get_output_layout();

        // right now output padding optimization is only available for bfyx format and data type = float32
        if (conv_layout.format != cldnn::format::bfyx &&
            conv_layout.format != cldnn::format::b_fs_yx_fsv16 &&
            conv_layout.format != cldnn::format::b_fs_zyx_fsv16 &&
            conv_layout.format != cldnn::format::bs_fs_yx_bsv16_fsv16 &&
            conv_layout.format != cldnn::format::b_fs_yx_fsv4 &&
            conv_layout.format != cldnn::format::fs_b_yx_fsv32 &&
            conv_layout.format != cldnn::format::b_fs_yx_32fp) {
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

        // Calculating input padding needed for convolution
        auto& filter_node = node.as<convolution>().weights(0);
        auto filter_prim = filter_node.get_primitive();

        layout filter_layout = filter_node.get_output_layout();

        // Compute initial required paddings for primitive used as input for convolution.
        auto input_offset = conv->input_offset;
        auto stride = conv->stride;
        auto dilation = conv->dilation;

        auto input_limit_x = input_offset.spatial[0] + (conv_layout.size.spatial[0] - 1) * stride.spatial[0] +
                             (filter_layout.size.spatial[0] - 1) * dilation.spatial[0] + 1;
        auto input_limit_y = input_offset.spatial[1] + (conv_layout.size.spatial[1] - 1) * stride.spatial[1] +
                             (filter_layout.size.spatial[1] - 1) * dilation.spatial[1] + 1;
        auto input_limit_z = input_offset.spatial[2] + (conv_layout.size.spatial[2] - 1) * stride.spatial[2] +
                             (filter_layout.size.spatial[2] - 1) * dilation.spatial[2] + 1;

        auto padding_begin_x = std::max(-input_offset.spatial[0], 0);
        auto padding_begin_y = std::max(-input_offset.spatial[1], 0);
        auto padding_begin_z = std::max(-input_offset.spatial[2], 0);
        auto padding_end_x = std::max(input_limit_x - prev_prim_output_layout.size.spatial[0], 0);
        auto padding_end_y = std::max(input_limit_y - prev_prim_output_layout.size.spatial[1], 0);
        auto padding_end_z = std::max(input_limit_z - prev_prim_output_layout.size.spatial[2], 0);

        // Adjust right padding, so entire buffer size in X dimension is properly aligned.
        // TODO: NOTE: Will be reenabled with next check-in once heuristic for line-aligned algorithm will be added.
        // auto needed_buffer_size_x = static_cast<cldnn::tensor::value_type>(
        //    round_up_to(left_padding + prev_prim_output_layout.size.spatial[0] + right_padding, 16));
        // right_padding = needed_buffer_size_x - left_padding - prev_prim_output_layout.size.spatial[0];

        cldnn::padding needed_padding({0, 0, padding_begin_x, padding_begin_y, padding_begin_z}, {0, 0, padding_end_x, padding_end_y, padding_end_z}, 0);
        needed_padding = padding::max(prev_prim_output_layout.data_padding, needed_padding);
        p.apply_needed_padding(node, conv_input_node, needed_padding);
    }

    for (auto& pair : p.nodes_map) {
        if (pair.second->type() != binary_convolution::type_id())
            continue;

        auto& node = pair.second->as<binary_convolution>();
        if (node.get_dependencies().empty())
            continue;

        auto conv = node.get_primitive();
        auto& conv_input_node = node.get_dependency(0);
        auto conv_layout = node.get_output_layout();

        // right now output padding optimization is only available for bfyx format and data type = float32
        if (conv_layout.format != cldnn::format::bfyx && conv_layout.format != cldnn::format::b_fs_yx_32fp)
            continue;

        // We shoudn't apply any padding to nodes which are marked as outputs or have type as data
        if (conv_input_node.is_output() || conv_input_node.is_type<data>())
            continue;

        // Calculating input padding needed for convolution
        auto& filter_node = node.as<binary_convolution>().weights(0);
        auto filter_prim = filter_node.get_primitive();

        layout filter_layout = filter_node.get_output_layout();

        // convolution have only one input primitive
        auto prev_prim_output_layout = conv_input_node.get_output_layout();

        // Compute initial required paddings for primitive used as input for convolution.
        auto input_offset = conv->input_offset;
        auto stride = conv->stride;
        auto dilation = conv->dilation;

        auto input_limit_x = input_offset.spatial[0] + (conv_layout.size.spatial[0] - 1) * stride.spatial[0] +
                             (filter_layout.size.spatial[0] - 1) * dilation.spatial[0] + 1;
        auto input_limit_y = input_offset.spatial[1] + (conv_layout.size.spatial[1] - 1) * stride.spatial[1] +
                             (filter_layout.size.spatial[1] - 1) * dilation.spatial[1] + 1;
        auto input_limit_z = input_offset.spatial[2] + (conv_layout.size.spatial[2] - 1) * stride.spatial[2] +
                             (filter_layout.size.spatial[2] - 1) * dilation.spatial[2] + 1;

        auto padding_begin_x = std::max(-input_offset.spatial[0], 0);
        auto padding_begin_y = std::max(-input_offset.spatial[1], 0);
        auto padding_begin_z = std::max(-input_offset.spatial[2], 0);
        auto padding_end_x = std::max(input_limit_x - prev_prim_output_layout.size.spatial[0], 0);
        auto padding_end_y = std::max(input_limit_y - prev_prim_output_layout.size.spatial[1], 0);
        auto padding_end_z = std::max(input_limit_z - prev_prim_output_layout.size.spatial[2], 0);

        cldnn::padding needed_padding({0, 0, padding_begin_x, padding_begin_y, padding_begin_z}, {0, 0, padding_end_x, padding_end_y, padding_end_z}, 0);
        needed_padding = padding::max(prev_prim_output_layout.data_padding, needed_padding);

        p.apply_needed_padding(node, conv_input_node, needed_padding);
    }
}
