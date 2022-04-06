// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "program_helpers.h"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include <algorithm>
#include <utility>
#include <vector>

namespace cldnn {
// helper function for merging the weights/biases buffers on cpu side for depthwise separable convolution optimization
void program_helpers::merge_buffers(engine& engine,
                                    program_node& node,
                                    const layout& target_layout,
                                    size_t begin_offset,
                                    size_t end_offset) {
    memory::ptr data_to_allocate = engine.allocate_memory(target_layout, false);
    auto& stream = node.get_program().get_stream();

    for (size_t i = begin_offset; i < end_offset; i++) {
        auto& weights = node.get_dependency(i).first->as<data>();
        mem_lock<char, mem_lock_type::read> src{weights.get_attached_memory_ptr(), stream};
        mem_lock<char, mem_lock_type::write> dst{data_to_allocate, stream};
        std::copy(src.begin(), src.end(), dst.begin() + (i - begin_offset) * src.size());
    }

    for (size_t i = 0; i < end_offset - begin_offset - 1; i++) node.remove_dependency(begin_offset + 1);

    auto& data_node = node.get_dependency(begin_offset).first->as<data>();
    data_node.attach_memory(data_to_allocate, false);
}

void program_helpers::reshape_deconvolution_weights(const std::vector<float> &deconv_weights,
    const int channels,
    const int kernel_width,
    const int kernel_height,
    const int scale_factor,
    std::vector<std::vector<std::vector<float> > >& subpixel_weights) {

    std::vector<std::vector<float> > weights(channels);

    int pad_zero_x = kernel_width % 2 == 0 ? 0 : 1;
    int pad_zero_y = kernel_height % 2 == 0 ? 0 : 1;

    // reshape 9x9 deconv weights, for example 32 9x9 deconv weights to 32 10x10 conv weights
    for (int f = 0; f < channels; ++f) {
        for (int kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
            for (int kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
                int index = f * kernel_width * kernel_height + kernel_y * kernel_width + kernel_x;
                weights[f].push_back(deconv_weights[index]);
            }
            if (pad_zero_x == 1) {    // pad with zero on x axis
                weights[f].push_back(0.f);
            }
        }
        if (pad_zero_y == 1) {    // pad a line on y axis with zero
            for (int kernel_x = 0; kernel_x < kernel_width + pad_zero_x; ++kernel_x) {
                weights[f].push_back(0.f);
            }
        }
    }

    // reshape 32 10x10 weights to 4 32 5x5 weights
    for (int s = 0; s < scale_factor*scale_factor; ++s) {
        subpixel_weights[s].resize(channels);
    }

    const int kernel_sz = kernel_width + pad_zero_x;

    auto get_row_index = [](int index, const int kernel_sz)->int {
        bool isRowEven = (index / (kernel_sz)) % 2 == 0 ? true : false;
        bool isColEven = (index % 2) == 0 ? true : false;
        int kernel_num = isRowEven ? (isColEven ? 0 : 1) : isColEven ? 2 : 3;
        return kernel_num;
    };

    int feature_num = static_cast<int>(weights.size());
    for (int f = 0; f < feature_num; ++f) {
        for (int i = 0; i < static_cast<int>(weights[f].size()); ++i) {
            int row = get_row_index(i, kernel_sz);
            subpixel_weights[row][f].push_back(weights[f][i]);
        }
    }

    // dump the weights for the shuffled kernel
    int subpixel_conv_num = static_cast<int>(subpixel_weights.size());
    for (int s = 0; s < subpixel_conv_num; ++s) {
        for (int row = 0; row < static_cast<int>(subpixel_weights[s].size()); ++row) {
            std::reverse(std::begin(subpixel_weights[s][row]), std::end(subpixel_weights[s][row]));
        }
    }
}

// helper function for getting target layout used in depthwise sep optimization
layout program_helpers::get_weights_layout(typed_program_node<cldnn::data>& data_node, int32_t split) {
    auto mem_layout = data_node.get_output_layout();

    return layout(mem_layout.data_type,
                  mem_layout.format,
                  {split * mem_layout.size.batch[0],
                   mem_layout.size.feature[0],
                   mem_layout.size.spatial[0],
                   mem_layout.size.spatial[1]});
}

// pair.first tells whether l1 and l2 are absolutely identical
// pair.second tells whether l1 and l2 can be reinterpreted to each other without need of reordering
// note: layouts can only be considered identical if data size described by both layouts match (so no data are genereted
// nor dropped) note: if layouts describe two buffers with different size, consider them not to be identical even if
// smaller buffer can be considered to hold subsequence of larger buffer,
//       this behavior is required to force buffer allocation for smaller buffer which, currently, should always be
//       performed
std::pair<bool, bool> program_helpers::are_layouts_identical(layout const& l1, layout const& l2) {
    const auto& l1_pad = l1.data_padding;
    const auto& l2_pad = l2.data_padding;
    auto offset_last_element_l1 = l1.get_linear_offset(l1.size - tensor{1});
    auto offset_last_element_l2 = l2.get_linear_offset(l2.size - tensor{1});
    if (l1 == l2)
        return {true, true};
    if (l1.data_type != l2.data_type)
        return {false, false};
    // Reorders between bfyx, bfzyx, bfwzyx can pe reinterpeted as reshape when
    // there is no padding and both hold same number of elements.
    if ((l1.format == format::bfyx || l1.format == format::bfzyx || l1.format == format::bfwzyx) &&
        (l2.format == format::bfyx || l2.format == format::bfzyx || l2.format == format::bfwzyx) && !l1_pad &&
        !l2_pad && l1.get_linear_size() == l2.get_linear_size())
        return {false, true};
    if (l1.size != l2.size)
        return {false, false};
    if (l1.get_linear_size() != l2.get_linear_size())
        return {false, false};

    auto check_format = [&l1, &l2](cldnn::format format) {
        return (l1.format == format && l2.format != format) ||
               (l2.format == format && l1.format != format);
    };

    if (check_format(format::b_fs_yx_fsv4) ||
        check_format(format::fs_b_yx_fsv32) ||
        check_format(format::b_fs_yx_fsv16) ||
        check_format(format::b_fs_yx_fsv32) ||
        check_format(format::b_fs_zyx_fsv32) ||
        check_format(format::b_fs_zyx_fsv16) ||
        check_format(format::bs_fs_yx_bsv4_fsv4) ||
        check_format(format::bs_fs_yx_bsv8_fsv4) ||
        check_format(format::bs_fs_yx_bsv8_fsv2) ||
        check_format(format::bs_fs_yx_bsv4_fsv2) ||
        check_format(format::bs_fs_yx_bsv32_fsv16) ||
        check_format(format::bs_fs_yx_bsv32_fsv32) ||
        check_format(format::bs_fs_yx_bsv16_fsv16) ||
        check_format(format::bs_fs_zyx_bsv16_fsv16))
        return {false, false};

    // If data is actually 1d along f and dense, the layouts are identical
    if (l1.data_type == l2.data_type && l1.size == l2.size && !l1_pad && !l2_pad && l1.size.batch[0] == 1 &&
        ((l1.format.spatial_num() == 2 && l1.size.spatial[0] == 1 && l1.size.spatial[1] == 1) ||
        ((l1.format.spatial_num() == 3 && l1.size.spatial[0] == 1 && l1.size.spatial[1] == 1 && l1.size.spatial[2] == 1))) &&
        (offset_last_element_l1 + 1 == l1.size.feature[0] && offset_last_element_l2 + 1 == l2.size.feature[0]))
        return {false, true};

    auto l1_pitch = l1.get_pitches();
    auto l2_pitch = l2.get_pitches();

    // ignore pitches which will never be used (for dims with size == 1)
    for (size_t i = 0; i < tensor_dim_max; ++i)
        if (l1.size.raw[i] == 1)
            l1_pitch.raw[i] = 0;
    for (size_t i = 0; i < tensor_dim_max; ++i)
        if (l2.size.raw[i] == 1)
            l2_pitch.raw[i] = 0;

    auto l1_offset = l1.get_linear_offset();
    auto l2_offset = l2.get_linear_offset();
    if (l1_pitch == l2_pitch && l1_offset == l2_offset)
        return {false, true};

    return {false, false};
}

// check if input and output layouts are identical to reuse memory in fused_ops of onednn
bool program_helpers::are_layouts_identical_for_onednn_sum_post_op(layout input_layout, layout output_layout) {
    if (input_layout.size == output_layout.size && input_layout.format == output_layout.format &&
        input_layout.data_padding == output_layout.data_padding &&
        data_type_traits::size_of(input_layout.data_type) == data_type_traits::size_of(output_layout.data_type))
        return true;

    return false;
}

bool program_helpers::needs_onednn_sum_post_op(const eltwise_node& n, layout input_layout) {
    auto output_layout = n.get_output_layout();
    if (n.get_primitive()->mode == eltwise_mode::sum &&
        (input_layout.size.spatial[0] > 1 || input_layout.size.spatial[1] > 1 || input_layout.size.batch[0] > 1)
        && output_layout.data_type == input_layout.data_type) {
        return true;
    }

    return false;
}




}  // namespace cldnn
