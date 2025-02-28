// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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

        if (desc.get_compression_zp_inputs_num() > 0)
            data_inputs_num -= 2;
    }
    return data_inputs_num;
}

inline ov::Dimension get_num_heads(const cldnn::layout& qkv, const std::vector<int64_t>& order) {
    return qkv.get_partial_shape()[order[1]];
}

}  // namespace ov::intel_gpu::ocl
