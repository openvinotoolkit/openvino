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

inline ov::Dimension get_num_heads(const cldnn::layout& qkv, const std::vector<int64_t>& order) {
    return qkv.get_partial_shape()[order[1]];
}

inline ChannelName get_transposed_channel(ChannelName c, const std::vector<int64_t>& order) {
    constexpr static std::array channels_order = {ChannelName::BATCH, ChannelName::FEATURE, ChannelName::Y, ChannelName::X};
    for (size_t i = 0; i < channels_order.size(); i++) {
        if (channels_order.at(i) == c) {
            size_t target_channel = i;
            auto transposed_channel = static_cast<size_t>(order[target_channel]);
            return channels_order.at(transposed_channel);
        }
    }
    return ChannelName::UNKNOWN;
}

inline bool is_prefill_stage(const RuntimeParams& params) {
    auto desc = params.typed_desc<cldnn::scaled_dot_product_attention>();
    const auto target_seq_len = extract_channel(get_transposed_channel(ChannelName::Y, desc->input_q_transpose_order), params.input_layouts[0]);

    return target_seq_len > 1;
}

}  // namespace ov::intel_gpu::ocl
