// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/dynamic_quantize_utils.cl"

// Dynamic quantization of the activations of the MoE expert GEMMs, independent of the weight type, with the contract of
// dynamic_quantize_gpu_opt: f16 [rows, K] to symmetric int8 [rows, K] per GROUP_SIZE run, with the dequantization scale
// of each group ([rows, K / GROUP_SIZE] f16).

#define SIMD      16
#define HALF_VEC  MAKE_VECTOR_TYPE(half, VEC_SIZE)
#define CHAR_VEC  MAKE_VECTOR_TYPE(char, VEC_SIZE)
#define VLOAD     CAT(vload, VEC_SIZE)
#define VSTORE    CAT(vstore, VEC_SIZE)

#if GROUP_SIZE != SIMD * VEC_SIZE
#   error "moe_3gemm_dyn_quan.cl - GROUP_SIZE has to be SIMD * VEC_SIZE"
#endif

// One subgroup quantizes one group; each lane owns VEC_SIZE consecutive values.
REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(moe_dyn_quan_quantize)(
    const __global half* input,
    __global char* quantized,
    __global half* scales)
{
    const uint group = (uint)get_global_id(0) / SIMD;
    const uint offset = group * GROUP_SIZE + get_sub_group_local_id() * VEC_SIZE;

    const HALF_VEC v = VLOAD(0, input + offset);
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
#undef CHAR_VEC
#undef VLOAD
#undef VSTORE
