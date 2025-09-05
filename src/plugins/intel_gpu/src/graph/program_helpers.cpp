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
            && !p_node.is_output()
            && !(dep_node.get_program().is_body_program() && dep_node.is_type<input_layout>())) {
            return add_fusing_type::sum;
        } else if (p_layout.get_tensor() == d_layout.get_tensor()) {
            return add_fusing_type::binary_per_tensor;
        }
    }

    return add_fusing_type::binary_per_oc;
}

int32_t onednn_add_fusing_helpers::get_reused_eltwmem_idx(const program_node& node) {
    if (node.get_preferred_impl_type() == impl_types::onednn) {
        for (auto& fused_op : node.get_fused_primitives()) {
            if (fused_op.is_type<eltwise>() && fused_op.deps.size() == 1) {
                // If it is first sum, reuse the buffer
                auto fusing_type = get_add_fusing_type(node, fused_op);
                if (fusing_type != add_fusing_type::sum)
                    continue;
                if (!fused_op.has_outer_dep())
                    continue;
                return fused_op.outer_dep_start_idx;
            }
        }
    }
    return -1;   // if -1, no reused input memory
}

post_op_dnnl_policy_type onednn_post_ops_fusing_helpers::get_post_op_dnnl_policy_type(const layout& data_layout, const layout& slope_layout) {
    // data_layout and slope_layout should have static shape
    OPENVINO_ASSERT(data_layout.is_static() && slope_layout.is_static(),
                    "[GPU] onednn_post_ops_fusing_helpers::get_post_op_dnnl_policy_type - data and slope layouts should be static");

    const auto& data_shape = data_layout.get_shape();
    const auto& slope_shape = slope_layout.get_shape();
    size_t data_rank = data_shape.size();
    size_t slope_rank = slope_shape.size();

    // Scalar slope (broadcast to all)
    if (slope_layout.count() == 1) {
        return post_op_dnnl_policy_type::COMMON;
    }

    // If slope is 1D, check for broadcasting along any axis
    if (slope_rank == 1) {
        for (size_t i = 0; i < data_rank; ++i) {
            if (slope_shape[0] == data_shape[i]) {
                switch (i) {
                    case 0: return post_op_dnnl_policy_type::PER_DIM_0;
                    case 1: return post_op_dnnl_policy_type::PER_DIM_1;
                    case 2: return post_op_dnnl_policy_type::PER_DIM_2;
                    case 3: return post_op_dnnl_policy_type::PER_DIM_3;
                    default: break; // Fallback to numpy rules for higher ranks
                }
            }
        }
    }

    // Check if slope is broadcastable to data (numpy rules)
    bool broadcastable = true;
    std::vector<int> broadcast_axes;
    size_t rank_diff = data_rank > slope_rank ? data_rank - slope_rank : 0;

    for (size_t i = 0; i < data_rank; ++i) {
        size_t data_dim = data_shape[i];
        size_t slope_dim = (i < rank_diff) ? 1 : slope_shape[i - rank_diff];

        if (slope_dim != 1 && slope_dim != data_dim) {
            broadcastable = false;
            break;
        }
        if (slope_dim != 1 && slope_dim == data_dim) {
            broadcast_axes.push_back(static_cast<int>(i));
        }
    }

    if (!broadcastable) {
        // If not broadcastable, fallback to POLICY_TOTAL to exception throw
        return post_op_dnnl_policy_type::POLICY_TOTAL;
    }

    // Sort axes for consistent multi-axis policy checks
    std::sort(broadcast_axes.begin(), broadcast_axes.end());

    // If fully matches data shape, treat as PER_TENSOR
    bool is_per_tensor = (broadcast_axes.size() == data_rank);
    if (is_per_tensor) {
        return post_op_dnnl_policy_type::PER_TENSOR;
    }

    // Single axis policies
    if (broadcast_axes.size() == 1) {
        int axis = broadcast_axes[0];
        switch (axis) {
            case 0: return post_op_dnnl_policy_type::PER_DIM_0;
            case 1: return post_op_dnnl_policy_type::PER_DIM_1;
            case 2: return post_op_dnnl_policy_type::PER_DIM_2;
            case 3: return post_op_dnnl_policy_type::PER_DIM_3;
            default: break;
        }
    }

    // Two axes policies
    if (broadcast_axes.size() == 2) {
        int axis0 = broadcast_axes[0];
        int axis1 = broadcast_axes[1];
        // Check for PER_DIM_01 (N, C)
        if (axis0 == 0 && axis1 == 1) {
            return post_op_dnnl_policy_type::PER_DIM_01;
        }
    }

    // If no case, fallback to POLICY_TOTAL to exception throw
    return post_op_dnnl_policy_type::POLICY_TOTAL;
}

int onednn_post_ops_fusing_helpers::get_prelu_mask(const layout& data_layout, const layout& slope_layout) {
    auto policy = get_post_op_dnnl_policy_type(data_layout, slope_layout);
    return get_default_mask(policy, data_layout.get_rank());
}

int onednn_post_ops_fusing_helpers::get_prelu_mask_from_layouts(const std::function<layout()>& get_output_layout,
                                                                const std::function<layout(int32_t)>& get_input_layout,
                                                                int32_t slope_input_idx) {
    auto data_layout = get_output_layout();
    auto slope_layout = get_input_layout(slope_input_idx);
    return get_prelu_mask(data_layout, slope_layout);
}

int onednn_post_ops_fusing_helpers::get_default_mask(post_op_dnnl_policy_type policy, int ndims) {
    switch (policy) {
        case post_op_dnnl_policy_type::PER_DIM_0: return (1 << 0);
        case post_op_dnnl_policy_type::PER_OC:
        case post_op_dnnl_policy_type::PER_DIM_1: return (1 << 1);
        case post_op_dnnl_policy_type::PER_OCIC:
        case post_op_dnnl_policy_type::PER_DIM_01: return (1 << 0) + (1 << 1);
        case post_op_dnnl_policy_type::PER_DIM_2: return (1 << 2);
        case post_op_dnnl_policy_type::PER_DIM_3: return (1 << 3);
        case post_op_dnnl_policy_type::PER_TENSOR:
            assert(ndims > 0 && ndims <= DNNL_MAX_NDIMS);
            return (1 << ndims) - 1;
        case post_op_dnnl_policy_type::COMMON: return 0;
        default: OPENVINO_THROW("Incorrect post_op_dnnl_policy_type");
    }
}
}  // namespace cldnn
