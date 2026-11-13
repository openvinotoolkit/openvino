// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/dynamic_quantize_utils.cl"

// Dynamic quantization of the activations of the MoE expert GEMMs, independent of the weight type: symmetric int8 per
// GROUP_SIZE run, as dynamic_quantize_gpu_opt, with the dequantization scale ([rows, groups] f16) of each group.
//  - QUANTIZE: the hidden states, once per token.
//  - QUANTIZE_SWIGLU: act(gate) * up, the input of the down projection.
//  - GATHER: copies the quantized hidden states of each token into its rows sorted by expert.

#if QUANTIZE || QUANTIZE_SWIGLU

#include "moe_gate_activation.cl"

#define SIMD      16
#define HALF_VEC  MAKE_VECTOR_TYPE(half, VEC_SIZE)
#define FLOAT_VEC MAKE_VECTOR_TYPE(float, VEC_SIZE)
#define CHAR_VEC  MAKE_VECTOR_TYPE(char, VEC_SIZE)
#define VLOAD     CAT(vload, VEC_SIZE)
#define VSTORE    CAT(vstore, VEC_SIZE)

#if GROUP_SIZE != SIMD * VEC_SIZE
#   error "moe_3gemm_dyn_quan.cl - GROUP_SIZE has to be SIMD * VEC_SIZE"
#endif

// One subgroup quantizes one group; each lane owns VEC_SIZE consecutive values.
REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(moe_dyn_quan_quantize)(
#if QUANTIZE_SWIGLU
    const __global half* up,
    const __global half* gate,
#else
    const __global half* input,
#endif
    __global char* quantized,
    __global half* scales)
{
    const uint group = (uint)get_global_id(0) / SIMD;
    const uint offset = group * GROUP_SIZE + get_sub_group_local_id() * VEC_SIZE;

#if QUANTIZE_SWIGLU
    const FLOAT_VEC g = CAT(convert_float, VEC_SIZE)(VLOAD(0, gate + offset));
    const FLOAT_VEC u = CAT(convert_float, VEC_SIZE)(VLOAD(0, up + offset));
    FLOAT_VEC act;
    unroll_for (uint i = 0; i < VEC_SIZE; ++i)
        act[i] = MOE_GATE_ACT(g[i]) * u[i];
    // The f16 activation paths store act(gate) * up as f16 before the down projection.
    const HALF_VEC v = CAT(convert_half, VEC_SIZE)(act);
#else
    const HALF_VEC v = VLOAD(0, input + offset);
#endif

    CHAR_VEC q;
    half scale;
    DQ_I8_SUBGROUP_QUANTIZE_N(VEC_SIZE, half, v, q, scale);
    VSTORE(q, 0, quantized + offset);
    if (get_sub_group_local_id() == 0) {
        scales[group] = scale;
    }
}

#undef SIMD
#undef HALF_VEC
#undef FLOAT_VEC
#undef CHAR_VEC
#undef VLOAD
#undef VSTORE

#elif GATHER

// Each gathered row copies the ROW_SIZE int8 values and the GROUPS scales of its token.
KERNEL(moe_dyn_quan_gather)(
    const __global char* quantized_tokens,
    const __global half* token_scales,
    const __global int* token_idx,
    __global char* quantized,
    __global half* scales)
{
    const uint i = (uint)get_global_id(0);
    const uint row = (uint)get_global_id(1);
    const int token = token_idx[row];
    if (token < 0)
        return;
    if (i < ROW_SIZE / 16) {
        const __global uint4* src = (const __global uint4*)(quantized_tokens + (size_t)token * ROW_SIZE);
        __global uint4* dst = (__global uint4*)(quantized + (size_t)row * ROW_SIZE);
        dst[i] = src[i];
    }
    if (i < GROUPS) {
        scales[(size_t)row * GROUPS + i] = token_scales[(size_t)token * GROUPS + i];
    }
}

#endif
