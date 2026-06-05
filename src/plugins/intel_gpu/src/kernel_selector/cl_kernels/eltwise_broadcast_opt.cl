// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

// Optimized broadcast elementwise multiply for the FLUX.2 transformer's hot multiplies
// (AdaLayerNorm modulation, RMSNorm channel scale, SwiGLU same-shape, activation scalar).
// Vectorized half8, broadcast index reduced to a single (compile-constant) PERIOD modulo,
// supports dynamic shapes (global = total/8 set at dispatch). Seed evolved by KernelFoundry;
// see examples_custom_task/openvino/ocl_broadcast_mul. Validate() guarantees: f16, planar,
// contiguous, a single MUL, output count % 8 == 0, and the broadcast operand's non-unit dims
// form an innermost-contiguous suffix so out[i] = a[i] * b[i % PERIOD] holds in flat memory.

#if FULL_IS_INPUT0
#    define A_PTR input0
#    define B_PTR input1
#    define A_OFFSET INPUT0_OFFSET
#    define B_OFFSET INPUT1_OFFSET
#else
#    define A_PTR input1
#    define B_PTR input0
#    define A_OFFSET INPUT1_OFFSET
#    define B_OFFSET INPUT0_OFFSET
#endif

KERNEL(eltwise_broadcast_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input0,
    const __global INPUT1_TYPE* input1,
    __global OUTPUT_TYPE* output)
{
    const uint base = (uint)get_global_id(0) * 8;

    const MAKE_VECTOR_TYPE(OUTPUT_TYPE, 8) va = vload8(0, A_PTR + A_OFFSET + base);
    MAKE_VECTOR_TYPE(OUTPUT_TYPE, 8) vb;

#if BCAST_SCALAR
    vb = (MAKE_VECTOR_TYPE(OUTPUT_TYPE, 8))(B_PTR[B_OFFSET]);
#elif BCAST_SAME_SHAPE
    vb = vload8(0, B_PTR + B_OFFSET + base);
#else
    // PERIOD is a compile-time constant and a multiple of 8; base is a multiple of 8, so the
    // 8-wide slice is contiguous within one broadcast period -> a single coalesced vload8.
    const uint c = base % PERIOD;
    vb = vload8(0, B_PTR + B_OFFSET + c);
#endif

    vstore8(va * vb, 0, output + OUTPUT_OFFSET + base);
}

#undef A_PTR
#undef B_PTR
#undef A_OFFSET
#undef B_OFFSET
