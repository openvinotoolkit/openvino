// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

kernel void clspv_bootstrap(global const float* input, global float* output, float increment) {
    const size_t index = get_global_id(0);
    output[index] = input[index] + increment;
}
