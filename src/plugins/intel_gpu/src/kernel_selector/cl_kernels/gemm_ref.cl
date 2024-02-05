// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

// Required JIT definitions:
// TRANSPOSE_INPUT0 [1/0]      - whether to tranpose first input.
// TRANSPOSE_INPUT1 [1/0]      - whether to tranpose second input.
// ACCUMULATOR_TYPE [DataType] - type used for intermediate results accumulation.

inline uint FUNC(get_input0_index_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#if INPUT0_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT0, b, f, w, z, y, x);
#else
#if INPUT0_DIMS == 4
    return INPUT0_GET_INDEX_SAFE(b, f, y, x);
#elif INPUT0_DIMS == 5
    return INPUT0_GET_INDEX_SAFE(b, f, z, y, x);
#elif INPUT0_DIMS == 6
    return INPUT0_GET_INDEX_SAFE(b, f, w, z, y, x);
#else
#   error gemm_ref.cl : Unsupported input 0 format
#endif
#endif
}

inline uint FUNC(get_input0_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
    return FUNC_CALL(get_input0_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT0_DIMS_ORDER);
}

inline uint FUNC(get_input1_index_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#if INPUT1_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT1, b, f, w, z, y, x);
#else
#if INPUT1_DIMS == 4
    return INPUT1_GET_INDEX_SAFE(b, f, y, x);
#elif INPUT1_DIMS == 5
    return INPUT1_GET_INDEX_SAFE(b, f, z, y, x);
#elif INPUT1_DIMS == 6
    return INPUT1_GET_INDEX_SAFE(b, f, w, z, y, x);
#else
#   error gemm_ref.cl : Unsupported input 1 format
#endif
#endif
}

inline uint FUNC(get_input1_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
    return FUNC_CALL(get_input1_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT1_DIMS_ORDER);
}

#if BEAM_TABLE_TERM
inline uint FUNC(get_bt_index_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#if BEAM_TABLE_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(BEAM_TABLE, b, f, w, z, y, x);
#else
#   error gemm_ref.cl : Unsupported beam table format
#endif
}

inline uint FUNC(get_bt_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#ifdef INDIRECT_INPUT0
    return FUNC_CALL(get_bt_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT0_DIMS_ORDER);
#else
    return FUNC_CALL(get_bt_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT1_DIMS_ORDER);
#endif
}

#endif // BEAM_TABLE_TERM

#ifdef BIAS_TERM
inline uint FUNC(get_input2_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#if INPUT2_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT2, b, f, w, z, y, x);
#else
#if INPUT2_DIMS == 4
    return INPUT2_GET_INDEX_SAFE(b, f, y, x);
#elif INPUT2_DIMS == 5
    return INPUT2_GET_INDEX_SAFE(b, f, z, y, x);
#elif INPUT2_DIMS == 6
    return INPUT2_GET_INDEX_SAFE(b, f, w, z, y, x);
#else
#   error gemm_ref.cl : Unsupported input 2 format
#endif
#endif
}
#endif // BIAS_TERM

#define INPUT0_SIZE_F INPUT0_FEATURE_NUM
#define INPUT0_SIZE_B INPUT0_BATCH_NUM

KERNEL(gemm_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input0,
    const __global INPUT1_TYPE* input1,
#ifdef BIAS_TERM
    const __global INPUT2_TYPE* input2,
#endif
#if BEAM_TABLE_TERM
    const __global BEAM_TABLE_TYPE* beam_table,
#endif
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const uint x = (uint)get_global_id(0);
    const uint y = (uint)get_global_id(1);

    uint bidx = get_global_id(2);
    const uint b = bidx % TR_OUTPUT_BATCH_NUM;
    bidx /= TR_OUTPUT_BATCH_NUM;
    const uint f = bidx % TR_OUTPUT_FEATURE_NUM;
    bidx /= TR_OUTPUT_FEATURE_NUM;
    const uint z = bidx % TR_OUTPUT_SIZE_Z;
    bidx /= TR_OUTPUT_SIZE_Z;
    const uint w = bidx % TR_OUTPUT_SIZE_W;

    const uint K = CAT(INPUT0_SIZE_, MATMUL_AXIS);

    ACCUMULATOR_TYPE acc = ACCUMULATOR_VAL_ZERO;

    for (uint ki = 0; ki < K; ++ki) {
        uint b0 = b;
        uint b1 = b;
        #if INDIRECT_INPUT0
            b0 = beam_table[FUNC_CALL(get_bt_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, ki)];
        #endif
        #if INDIRECT_INPUT1
            b1 = beam_table[FUNC_CALL(get_bt_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, ki, x)];
        #endif

        uint in0_idx = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b0, f, w, z, y, ki);
        uint in1_idx = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b1, f, w, z, ki, x);

        ACCUMULATOR_TYPE val0 = TO_ACCUMULATOR_TYPE(input0[in0_idx]);
        ACCUMULATOR_TYPE val1 = TO_ACCUMULATOR_TYPE(input1[in1_idx]);

        acc += val0 * val1;
    }

    acc = TO_ACCUMULATOR_TYPE(ALPHA) * acc;

#ifdef BIAS_TERM
    {
        uint in2_idx = FUNC_CALL(get_input2_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x);
        ACCUMULATOR_TYPE val2 = TO_ACCUMULATOR_TYPE(input2[in2_idx]);

        acc += TO_ACCUMULATOR_TYPE(BETA) * val2;
    }
#endif

    const uint dst_index = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR TR_B, TR_F, TR_W, TR_Z, TR_Y, TR_X);

    ACTIVATION_TYPE dequantized = TO_ACTIVATION_TYPE(acc);

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE res = FUSED_OPS_RESULT;
    output[dst_index] = res;
#else
    output[dst_index] = dequantized;
#endif
}

#undef INPUT0_SIZE_F
#undef INPUT0_SIZE_B