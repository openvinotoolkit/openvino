// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "include/common.cl"
#include "include/fetch.cl"

// Required JIT definitions:
// TRANSPOSE_INPUT0 [1/0]      - whether to tranpose first input.
// TRANSPOSE_INPUT1 [1/0]      - whether to tranpose second input.
// ACCUMULATOR_TYPE [DataType] - type used for intermediate results accumulation.

inline uint FUNC(get_input0_index_nt)(uint b, uint f, uint w, uint z, uint y, uint x) {
#if INPUT0_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT0, b, f, w, z, y, x);
#else
#   error gemm_ref.cl : Unsupported input 0 format
#endif
}

inline uint FUNC(get_input0_index)(uint b, uint f, uint w, uint z, uint y, uint x) {
#if !TRANSPOSE_INPUT0
    return FUNC_CALL(get_input0_index_nt)(b, f, w, z, y, x);
#else
    return FUNC_CALL(get_input0_index_nt)(b, f, w, z, x, y);
#endif
}

inline uint FUNC(get_input1_index_nt)(uint b, uint f, uint w, uint z, uint y, uint x) {
#if INPUT1_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT1, b, f, w, z, y, x);
#else
#   error gemm_ref.cl : Unsupported input 1 format
#endif
}

inline uint FUNC(get_input1_index)(uint b, uint f, uint w, uint z, uint y, uint x) {
#if !TRANSPOSE_INPUT1
    return FUNC_CALL(get_input1_index_nt)(b, f, w, z, y, x);
#else
    return FUNC_CALL(get_input1_index_nt)(b, f, w, z, x, y);
#endif
}

#ifdef INPUT2_TYPE
inline uint FUNC(get_input2_index)(uint b, uint f, uint w, uint z, uint y, uint x) {
#if INPUT2_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT2, b, f, w, z, y, x);
#else
#   error gemm_ref.cl : Unsupported input 2 format
#endif
}
#endif // INPUT2_TYPE

inline uint FUNC(get_output_index)(uint b, uint f, uint w, uint z, uint y, uint x) {
#if OUTPUT_SIMPLE
    return GET_DATA_INDEX_6D(OUTPUT, b, f, w, z, y, x);
#else
#   error gemm_ref.cl : Unsupported output format
#endif
}

KERNEL(gemm_ref)(
    const __global INPUT0_TYPE* input0,
    const __global INPUT1_TYPE* input1,
#ifdef INPUT2_TYPE
    const __global INPUT2_TYPE* input2,
#endif
    __global OUTPUT_TYPE* output)
{
    const uint x = (uint)get_global_id(0);
    const uint y = (uint)get_global_id(1);

    uint bidx = get_global_id(2);
    const uint b = bidx % OUTPUT_BATCH_NUM;
    bidx /= OUTPUT_BATCH_NUM;
    const uint f = bidx % OUTPUT_FEATURE_NUM;
    bidx /= OUTPUT_FEATURE_NUM;
    const uint z = bidx % OUTPUT_SIZE_Z;
    bidx /= OUTPUT_SIZE_Z;
    const uint w = bidx % OUTPUT_SIZE_W;

#if !TRANSPOSE_INPUT0
    const uint K = INPUT0_SIZE_X;
#else
    const uint K = INPUT0_SIZE_Y;
#endif

    ACCUMULATOR_TYPE acc = ACCUMULATOR_VAL_ZERO;

    for (uint ki = 0; ki < K; ++ki) {
        uint in0_idx = FUNC_CALL(get_input0_index)(b, f, w, z, y, ki);
        uint in1_idx = FUNC_CALL(get_input1_index)(b, f, w, z, ki, x);

        ACCUMULATOR_TYPE val0 = TO_ACCUMULATOR_TYPE(input0[in0_idx]);
        ACCUMULATOR_TYPE val1 = TO_ACCUMULATOR_TYPE(input1[in1_idx]);

        acc = fma(val0, val1, acc);
    }

    acc = TO_ACCUMULATOR_TYPE(ALPHA) * acc;

#ifdef INPUT2_TYPE
    {
        uint in2_idx = FUNC_CALL(get_input2_index)(b, f, w, z, y, x);
        ACCUMULATOR_TYPE val2 = TO_ACCUMULATOR_TYPE(input2[in2_idx]);

        acc = fma(TO_ACCUMULATOR_TYPE(BETA), val2, acc);
    }
#endif

    const uint out_idx = FUNC_CALL(get_output_index)(b, f, w, z, y, x);

    output[out_idx] = TO_OUTPUT_TYPE_SAT(acc);
}
