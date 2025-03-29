// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "program_helpers.h"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "pooling_inst.h"
#include <algorithm>
#include <utility>
#include <vector>
#include <sstream>

namespace cldnn {
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

bool onednn_add_fusing_helpers::is_full_tensor(const layout& l) {
    if (l.spatial(0) > 1 || l.spatial(1) > 1 || (l.get_spatial_rank() == 3 && l.spatial(2) > 1)
        || l.batch() > 1) {
        return true;
    }
    return false;
}

void onednn_add_fusing_helpers::for_eltwise(
    const program_node& node, eltwise_mode mode,
    std::function<void(const program_node& p_node,
                    const fused_primitive_desc& desc)> func) {
    for (auto& fo : node.get_fused_primitives()) {
        if (fo.is_type<eltwise>() && fo.typed_desc<eltwise>()->mode == mode) {
            func(node, fo);
        }
    }
}

static bool is_direct_ancestor(const program_node& child, const program_node& target) {
    // is_direct_ancestor function is added to detect pattern like below from get_add_fusing_type.
    // It is necessary to use onednn sum post operation in such case for better performance.
    // In such case, 'A' can have two connections.
    //   ┌───────┐
    //   │   A   │
    //   └──┬──┬─┘
    //      │  └────────────┐
    //      │           ┌───┴───┐
    //      │post_op    │   B   │
    //      │           └───┬───┘
    //      │  ┌────────────┘
    //   ┌──┴──┴─┐
    //   │   C   │
    //   └───────┘
    if (target.get_users().size() != 2)
        return false;

    // Limit the iteration depth to 5 for performance reason
    auto iter = &child;
    for (int i = 0; i < 5; i++) {
        if (iter == &target)
            return true;
        if (iter->get_dependencies().size() == 0)
            break;
        iter = &iter->get_dependency(0);
    }
    return false;
}

add_fusing_type onednn_add_fusing_helpers::get_add_fusing_type(
    const program_node& p_node, const fused_primitive_desc& desc) {
    if (!desc.is_type<eltwise>()) {
        return add_fusing_type::not_supported;
    }
    if (desc.typed_desc<eltwise>()->mode != eltwise_mode::sum) {
        return add_fusing_type::not_supported;
    }
    if (!desc.has_outer_dep()) {
        return add_fusing_type::not_supported;
    }
    auto& dep_node = p_node.get_dependency(desc.outer_dep_start_idx);
    auto p_layout = p_node.get_output_layout();
    auto d_layout = dep_node.get_output_layout();

    if (p_node.is_dynamic() || dep_node.is_dynamic()) {
        return add_fusing_type::not_supported;
    }

    if (is_full_tensor(p_layout) && is_full_tensor(d_layout)) {
        if (data_type_traits::size_of(p_layout.data_type) == data_type_traits::size_of(d_layout.data_type)
            && p_layout.format == d_layout.format && p_layout.get_tensor() == d_layout.get_tensor()
            && p_layout.data_padding == d_layout.data_padding
            && (dep_node.get_users().size() == 1 || is_direct_ancestor(p_node, dep_node))
            && !dep_node.is_constant()
            && !p_node.is_type<pooling>()
            && !(dep_node.get_program().is_body_program() && dep_node.is_type<input_layout>())) {
            return add_fusing_type::sum;
        } else if (p_layout.get_tensor() == d_layout.get_tensor()) {
            return add_fusing_type::binary_per_tensor;
        }
    }

    return add_fusing_type::binary_per_oc;
}


}  // namespace cldnn
