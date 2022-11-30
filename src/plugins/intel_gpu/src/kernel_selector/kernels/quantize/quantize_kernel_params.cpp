// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "quantize_kernel_params.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {

size_t quantize_params::hash() const {
    auto seed = base_params::hash();
    seed = hash_combine(seed, levels);
    seed = hash_combine(seed, packed_binary_output);
    seed = hash_combine(seed, scale_shift_opt);

    seed = hash_combine(seed, has_post_scale);
    seed = hash_combine(seed, has_post_shift);
    seed = hash_combine(seed, has_pre_shift);

    seed = hash_combine(seed, has_clamp);
    seed = hash_combine(seed, has_min_clamp);
    seed = hash_combine(seed, has_max_clamp);

    seed = hash_combine(seed, per_tensor_input_range);
    seed = hash_combine(seed, per_tensor_input_scale);
    seed = hash_combine(seed, per_tensor_input_shift);

    seed = hash_combine(seed, per_tensor_output_range);
    seed = hash_combine(seed, per_tensor_output_scale);
    seed = hash_combine(seed, per_tensor_output_shift);

    seed = hash_combine(seed, in_lo);
    seed = hash_combine(seed, in_hi);
    seed = hash_combine(seed, in_scale);
    seed = hash_combine(seed, in_shift);

    seed = hash_combine(seed, out_lo);
    seed = hash_combine(seed, out_hi);
    seed = hash_combine(seed, out_scale);
    seed = hash_combine(seed, out_shift);
    return seed;
}
}  // namespace kernel_selector
