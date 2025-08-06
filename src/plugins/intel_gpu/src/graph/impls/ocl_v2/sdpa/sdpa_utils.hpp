// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_utils/dispatch_utils.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "scaled_dot_product_attention_inst.h"

namespace ov::intel_gpu::ocl {

// Data inputs are QKV, mask and scale. Beam table and quantization params are not considered.
inline size_t get_data_inputs_num(const cldnn::scaled_dot_product_attention& desc) {
    size_t data_inputs_num = desc.input_size();
    if (desc.indirect_axis != -1) {
        data_inputs_num--;
    }

    if (desc.is_kv_compressed) {
        data_inputs_num -= 2;
        if (desc.get_compression_zp_inputs_num() > 0) {
            data_inputs_num -= 2;
        }
    }
    return data_inputs_num;
}

inline size_t get_key_cache_id(const cldnn::scaled_dot_product_attention& desc) {
    size_t key_cache_id = desc.input_size();

    if (!desc.is_kv_compressed) {
        return -1;
    }

    if (desc.indirect_axis != -1)
        key_cache_id -= 1;  // beam_table
    if (desc.get_compression_zp_inputs_num() > 0) {
        key_cache_id -= 4;
    } else {
        key_cache_id -= 2;  // Scales
    }

    return key_cache_id;
}

inline size_t get_value_cache_id(const cldnn::scaled_dot_product_attention& desc) {
    size_t value_cache_id = desc.input_size();

    if (!desc.is_kv_compressed) {
        return -1;
    }

    if (desc.indirect_axis != -1)
        value_cache_id -= 1;  // beam_table
    if (desc.get_compression_zp_inputs_num() > 0) {
        value_cache_id -= 3;  // Scales and zp
    } else {
        value_cache_id -= 1;  // Scales
    }

    return value_cache_id;
}

inline std::vector<int64_t> extend_order_in_num_heads_dim(const std::vector<int64_t>& order, size_t rank = 4) {
    if (order.size() == rank) {
        return order;
    }

    std::vector<int64_t> extended_order(rank, 0);
    const size_t num_heads_dim = 1;
    // For 3D dimension, extend it to 4D by adding 1 for num_heads_dim
    for (size_t i = 0, j = 0; i < rank; ++i) {
        if (i == num_heads_dim) {
            extended_order[num_heads_dim] = 1;
        } else {
            extended_order[i] = (static_cast<size_t>(order[j]) < num_heads_dim) ? order[j] : order[j] + 1;
            j++;
        }
    }
    return extended_order;
}

inline int64_t get_batch_size(const cldnn::layout& qkv, const std::vector<int64_t>& order) {
    auto& dim = qkv.get_partial_shape()[order[0]];
    if (dim.is_dynamic())
        return -1;
    else
        return dim.get_length();
}

inline int64_t get_num_heads(const cldnn::layout& qkv, const std::vector<int64_t>& order) {
    // 4D - BHLS
    // 3D - BLS and H=1
    const auto order_rank = order.size();
    if (order_rank == 3) {
        return 1;
    } else {
        auto& dim = qkv.get_partial_shape()[order[order_rank - 3]];
        if (dim.is_dynamic())
            return -1;
        else
            return dim.get_length();
    }
}

inline int64_t get_head_size(const cldnn::layout& qkv, const std::vector<int64_t>& order) {
    const auto order_rank = order.size();
    auto& dim = qkv.get_partial_shape()[order[order_rank - 1]];
    if (dim.is_dynamic())
        return -1;
    else
        return dim.get_length();
}

inline int64_t get_seq_length(const cldnn::layout& qkv, const std::vector<int64_t>& order) {
    const auto order_rank = order.size();
    auto& dim = qkv.get_partial_shape()[order[order_rank - 2]];
    if (dim.is_dynamic()) {
        return -1;
    } else {
        return dim.get_length();
    }
}

inline ChannelName get_transposed_channel(ChannelName c, const std::vector<int64_t>& order) {
    if (order.size() == 4) {
        constexpr std::array channels_order = {ChannelName::BATCH, ChannelName::FEATURE, ChannelName::Y, ChannelName::X};
        for (size_t i = 0; i < channels_order.size(); i++) {
            if (channels_order.at(i) == c) {
                size_t target_channel = i;
                auto transposed_channel = static_cast<size_t>(order[target_channel]);
                return channels_order.at(transposed_channel);
            }
        }
    } else if (order.size() == 3) {
        constexpr std::array channels_order = {ChannelName::BATCH, ChannelName::FEATURE, ChannelName::Y};
        for (size_t i = 0; i < channels_order.size(); i++) {
            if (channels_order.at(i) == c) {
                size_t target_channel = i;
                auto transposed_channel = static_cast<size_t>(order[target_channel]);
                return channels_order.at(transposed_channel);
            }
        }
    }
    return ChannelName::UNKNOWN;
}

inline bool is_prefill_stage(const RuntimeParams& params) {
    auto desc = params.typed_desc<cldnn::scaled_dot_product_attention>();
    const auto target_seq_len = extract_channel(get_transposed_channel(ChannelName::Y, desc->input_q_transpose_order), params.input_layouts[0]);
    // const auto target_seq_len_0 = get_seq_length(params.get_input_layout(0), desc->input_q_transpose_order);

    return target_seq_len > 1;
}

inline bool unaligned_head_size(const RuntimeParams& params) {
    auto desc = params.typed_desc<cldnn::scaled_dot_product_attention>();
    constexpr size_t subgroup_size = 16;
    const auto k_head_size = get_head_size(params.get_input_layout(1), desc->input_k_transpose_order);
    const auto v_head_size = get_head_size(params.get_input_layout(2), desc->input_v_transpose_order);
    return (k_head_size % subgroup_size != 0) || (v_head_size % subgroup_size != 0);
}

}  // namespace ov::intel_gpu::ocl
