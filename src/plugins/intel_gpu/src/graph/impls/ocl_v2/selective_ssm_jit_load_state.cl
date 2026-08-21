// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Kernel-body fragment. The including kernel provides
// SSM_JIT_LOAD_STATE(p, state_element) for its state-buffer layout.
#ifndef SSM_JIT_LOAD_STATE
#    error "SelectiveSSM state load expression is not defined"
#endif

#pragma unroll
for (uint p_offset = 0; p_offset < SSM_HEAD_DIM_BLOCK; ++p_offset) {
    const uint p = p_base + p_offset;
#pragma unroll
    for (uint step = 0; step < SSM_STATE_ITERATIONS; ++step) {
        const uint state_element = step * SSM_SUBGROUP_SIZE + lane;
        if (p < SSM_HEAD_DIM && state_element < SSM_STATE_SIZE)
            SSM_STATE_AT(p_offset, step, state_element) = SSM_JIT_LOAD_STATE(p, state_element);
    }
}
