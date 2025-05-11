// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_weights.cl"
#include "include/mmad.cl"

#define INPUT_PACKED_TYPE_8     CAT(INPUT_PACKED_TYPE, 8)
#define FILTER_PACKED_TYPE_8    CAT(FILTER_PACKED_TYPE, 8)
#define INPUT_PACKED_TYPE_VEC   CAT(INPUT_PACKED_TYPE, SUB_GROUP_SIZE)
#define FILTER_PACKED_TYPE_VEC  CAT(FILTER_PACKED_TYPE, SUB_GROUP_SIZE)

#define BLOCK_READ(ptr)         _sub_group_block_read((const __global uint*)(ptr))
#define BLOCK_READ_8(ptr)       _sub_group_block_read8((const __global uint*)(ptr))

#define MMAD                    CAT(MMAD_, SUB_GROUP_SIZE)

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
KERNEL(fully_connected_gpu_MMAD)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const uint lid0 = (uint)get_local_id(0);
    const uint feature_per_wg = (uint)get_local_size(0) / SLM_DIV_FACTOR;
    const uint feature = (uint)get_group_id(0) * feature_per_wg + (uint)get_global_id(0) % feature_per_wg;
    const uint feature_block = lid0 / feature_per_wg;
    const uint batch = (uint)get_global_id(1);
    const uint skip_f = (uint)get_global_id(2);

    int dotProd = 0;

    const uint filter_offset = FILTER_GET_OFFSET(feature);
#if INPUT0_DIMS == 5
    const uint input_offset = INPUT0_GET_INDEX(batch, 0, 0, 0, 0);
#else
    const uint input_offset = INPUT0_GET_INDEX(batch, skip_f, 0, 0);
#endif

#if SLM_DIV_FACTOR > 1
    __local int partial_summ[WORK_GROUP_SIZE];
#endif

#if SPATIAL_MAJOR

#if FULL_UNROLL_FACTOR < 2
    for (uint k = feature_block * FULL_UNROLL_FACTOR; k < (feature_block + 1) * FULL_UNROLL_FACTOR; ++k)
#elif UNROLL_FACTOR == FULL_UNROLL_FACTOR
    uint k = feature_block * FULL_UNROLL_FACTOR;
#else
    for (uint k = feature_block * FULL_UNROLL_FACTOR; k + UNROLL_FACTOR <= (feature_block + 1) * FULL_UNROLL_FACTOR; k += UNROLL_FACTOR)
#endif // FULL_UNROLL_FACTOR < 2
    {
#if !SPLIT_SPATIAL
        for (uint spatial = 0; spatial < FILTER_SPATIAL_SIZE; ++spatial) {
#else
        for (uint zi = 0; zi < FILTER_SIZE_Z; ++zi)
        for (uint yi = 0; yi < FILTER_SIZE_Y; ++yi)
        for (uint xi = 0; xi < FILTER_SIZE_X; ++xi) {
            const uint spatial = xi + yi * FILTER_SIZE_X + zi * FILTER_SIZE_X * FILTER_SIZE_Y;
#endif // !SPLIT_SPATIAL

#else  // SPATIAL_MAJOR

#if !SPLIT_SPATIAL
    for (uint spatial = 0; spatial < FILTER_SPATIAL_SIZE; ++spatial) {
#else
    for (uint zi = 0; zi < FILTER_SIZE_Z; ++zi)
    for (uint yi = 0; yi < FILTER_SIZE_Y; ++yi)
    for (uint xi = 0; xi < FILTER_SIZE_X; ++xi) {
        const uint spatial = xi + yi * FILTER_SIZE_X + zi * FILTER_SIZE_X * FILTER_SIZE_Y;
#endif // !SPLIT_SPATIAL

#if FULL_UNROLL_FACTOR < 2
        for (uint k = feature_block * FULL_UNROLL_FACTOR; k < (feature_block + 1) * FULL_UNROLL_FACTOR; ++k)
#elif UNROLL_FACTOR == FULL_UNROLL_FACTOR
        uint k = feature_block * FULL_UNROLL_FACTOR;
#else
        for (uint k = feature_block * FULL_UNROLL_FACTOR; k + UNROLL_FACTOR <= (feature_block + 1) * FULL_UNROLL_FACTOR; k += UNROLL_FACTOR)
#endif // FULL_UNROLL_FACTOR < 2
        {
#endif // SPATIAL_MAJOR

#if !SPLIT_SPATIAL
            uint input_idx = input_offset + spatial * MMAD_INPUT_SPATIAL_PITCH + k * MMAD_INPUT_FBLOCK_PITCH;
#else
            uint input_idx = input_offset + k * MMAD_INPUT_FBLOCK_PITCH + zi * MMAD_INPUT_Z_PITCH + yi * MMAD_INPUT_Y_PITCH + xi * MMAD_INPUT_X_PITCH;
#endif // !SPLIT_SPATIAL
            uint filter_idx = filter_offset + spatial * MMAD_FILTER_SPATIAL_PITCH + k * MMAD_FILTER_FBLOCK_PITCH;

#if UNROLL_FACTOR < 2
            INPUT_PACKED_TYPE input_data = AS_TYPE(INPUT_PACKED_TYPE, BLOCK_READ(input + input_idx));
            INPUT_PACKED_TYPE_VEC activations;

            activations.s0 = sub_group_broadcast(input_data, 0);
            activations.s1 = sub_group_broadcast(input_data, 1);
            activations.s2 = sub_group_broadcast(input_data, 2);
            activations.s3 = sub_group_broadcast(input_data, 3);
            activations.s4 = sub_group_broadcast(input_data, 4);
            activations.s5 = sub_group_broadcast(input_data, 5);
            activations.s6 = sub_group_broadcast(input_data, 6);
            activations.s7 = sub_group_broadcast(input_data, 7);
#if SUB_GROUP_SIZE == 16
            activations.s8 = sub_group_broadcast(input_data, 8);
            activations.s9 = sub_group_broadcast(input_data, 9);
            activations.sa = sub_group_broadcast(input_data, 0xa);
            activations.sb = sub_group_broadcast(input_data, 0xb);
            activations.sc = sub_group_broadcast(input_data, 0xc);
            activations.sd = sub_group_broadcast(input_data, 0xd);
            activations.se = sub_group_broadcast(input_data, 0xe);
            activations.sf = sub_group_broadcast(input_data, 0xf);
#endif // SUB_GROUP_SIZE == 16

            FILTER_PACKED_TYPE_VEC weights_data;
#if SUB_GROUP_SIZE == 8
            weights_data = AS_TYPE(FILTER_PACKED_TYPE_8, BLOCK_READ_8(weights + filter_idx));
#else
            weights_data.lo = AS_TYPE(FILTER_PACKED_TYPE_8, BLOCK_READ_8(weights + filter_idx));
            weights_data.hi = AS_TYPE(FILTER_PACKED_TYPE_8, BLOCK_READ_8(weights + filter_idx + SUB_GROUP_SIZE * 8 * 4));
#endif // SUB_GROUP_SIZE == 8

            dotProd = MMAD(activations, weights_data, dotProd);
#else // UNROLL_FACTOR < 2
            INPUT_PACKED_TYPE input_data[UNROLL_FACTOR];
            FILTER_PACKED_TYPE_VEC weights_data[UNROLL_FACTOR];

            unroll_for (uint kb = 0; kb < UNROLL_FACTOR; kb++) {
                input_data[kb] = AS_TYPE(INPUT_PACKED_TYPE, BLOCK_READ(input + input_idx + kb * MMAD_INPUT_FBLOCK_PITCH));
#if SUB_GROUP_SIZE == 8
                weights_data[kb] = AS_TYPE(FILTER_PACKED_TYPE_8, BLOCK_READ_8(weights + filter_idx + kb * MMAD_FILTER_FBLOCK_PITCH));
#else
                weights_data[kb].lo = AS_TYPE(FILTER_PACKED_TYPE_8, BLOCK_READ_8(weights + filter_idx + kb * MMAD_FILTER_FBLOCK_PITCH));
                weights_data[kb].hi = AS_TYPE(FILTER_PACKED_TYPE_8, BLOCK_READ_8(weights + filter_idx + SUB_GROUP_SIZE * 32 + kb * MMAD_FILTER_FBLOCK_PITCH));
#endif // SUB_GROUP_SIZE
            }

            unroll_for (uint kb = 0; kb < UNROLL_FACTOR; kb++) {
                INPUT_PACKED_TYPE_VEC in;

                in.s0 = sub_group_broadcast(input_data[kb], 0);
                in.s1 = sub_group_broadcast(input_data[kb], 1);
                in.s2 = sub_group_broadcast(input_data[kb], 2);
                in.s3 = sub_group_broadcast(input_data[kb], 3);
                in.s4 = sub_group_broadcast(input_data[kb], 4);
                in.s5 = sub_group_broadcast(input_data[kb], 5);
                in.s6 = sub_group_broadcast(input_data[kb], 6);
                in.s7 = sub_group_broadcast(input_data[kb], 7);
#if SUB_GROUP_SIZE == 16
                in.s8 = sub_group_broadcast(input_data[kb], 8);
                in.s9 = sub_group_broadcast(input_data[kb], 9);
                in.sa = sub_group_broadcast(input_data[kb], 0xa);
                in.sb = sub_group_broadcast(input_data[kb], 0xb);
                in.sc = sub_group_broadcast(input_data[kb], 0xc);
                in.sd = sub_group_broadcast(input_data[kb], 0xd);
                in.se = sub_group_broadcast(input_data[kb], 0xe);
                in.sf = sub_group_broadcast(input_data[kb], 0xf);
#endif // SUB_GROUP_SIZE == 16
                dotProd = MMAD(in, weights_data[kb], dotProd);
            }
#endif // UNROLL_FACTOR < 2
        }
    }

#if SLM_DIV_FACTOR > 1
    partial_summ[lid0] = dotProd;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (feature_block == 0) {
        unroll_for(uint i = 1; i < SLM_DIV_FACTOR; i++)
            dotProd += partial_summ[lid0 % feature_per_wg + i * feature_per_wg];
#endif // SLM_DIV_FACTOR > 1

#if HAS_FEATURE_LEFTOVERS
        const uint sglid = get_sub_group_local_id();
#if SPATIAL_MAJOR
#if !SPLIT_SPATIAL
        for (uint spatial = 0; spatial < FILTER_SPATIAL_SIZE; ++spatial) {
#else
        for (uint zi = 0; zi < FILTER_SIZE_Z; ++zi)
        for (uint yi = 0; yi < FILTER_SIZE_Y; ++yi)
        for (uint xi = 0; xi < FILTER_SIZE_X; ++xi) {
            const uint spatial = xi + yi * FILTER_SIZE_X + zi * FILTER_SIZE_X * FILTER_SIZE_Y;
#endif  // !SPLIT_SPATIAL

#else  // SPATIAL_MAJOR

#if !SPLIT_SPATIAL
    for (uint spatial = 0; spatial < FILTER_SPATIAL_SIZE; ++spatial) {
#else  // !SPLIT_SPATIAL
    for (uint zi = 0; zi < FILTER_SIZE_Z; ++zi)
        for (uint yi = 0; yi < FILTER_SIZE_Y; ++yi)
        for (uint xi = 0; xi < FILTER_SIZE_X; ++xi) {
            const uint spatial = xi + yi * FILTER_SIZE_X + zi * FILTER_SIZE_X * FILTER_SIZE_Y;
#endif  // !SPLIT_SPATIAL

#endif  // SPATIAL_MAJOR

#if !SPLIT_SPATIAL
            uint input_idx = input_offset + spatial * MMAD_INPUT_SPATIAL_PITCH + FEATURE_BLOCKS_COUNT * MMAD_INPUT_FBLOCK_PITCH;
#else
            uint input_idx = input_offset + FEATURE_BLOCKS_COUNT * MMAD_INPUT_FBLOCK_PITCH +
                             zi * MMAD_INPUT_Z_PITCH + yi * MMAD_INPUT_Y_PITCH + xi * MMAD_INPUT_X_PITCH;
#endif // !SPLIT_SPATIAL
            uint filter_idx = filter_offset + spatial * MMAD_FILTER_SPATIAL_PITCH + FEATURE_BLOCKS_COUNT * MMAD_FILTER_FBLOCK_PITCH;

            MAKE_VECTOR_TYPE(INPUT0_TYPE, 4) input_data_u = (0, 0, 0, 0);
            for (uint i = 0; i < 4; i++) {
                if (FEATURE_BLOCKS_COUNT * SUB_GROUP_SIZE * 4 + sglid * 4 + i < IN_FEATURE_NUM) {
                    input_data_u[i] = input[input_idx + (sglid * 4 + i) * FEATURE_PITCH];
                }
            }
            INPUT_PACKED_TYPE input_data = AS_TYPE(INPUT_PACKED_TYPE, input_data_u);

            INPUT_PACKED_TYPE_VEC activations;

            activations.s0 = sub_group_broadcast(input_data, 0);
            activations.s1 = sub_group_broadcast(input_data, 1);
            activations.s2 = sub_group_broadcast(input_data, 2);
            activations.s3 = sub_group_broadcast(input_data, 3);
            activations.s4 = sub_group_broadcast(input_data, 4);
            activations.s5 = sub_group_broadcast(input_data, 5);
            activations.s6 = sub_group_broadcast(input_data, 6);
            activations.s7 = sub_group_broadcast(input_data, 7);
#if SUB_GROUP_SIZE == 16
            activations.s8 = sub_group_broadcast(input_data, 8);
            activations.s9 = sub_group_broadcast(input_data, 9);
            activations.sa = sub_group_broadcast(input_data, 0xa);
            activations.sb = sub_group_broadcast(input_data, 0xb);
            activations.sc = sub_group_broadcast(input_data, 0xc);
            activations.sd = sub_group_broadcast(input_data, 0xd);
            activations.se = sub_group_broadcast(input_data, 0xe);
            activations.sf = sub_group_broadcast(input_data, 0xf);
#endif // SUB_GROUP_SIZE == 16

            FILTER_PACKED_TYPE_VEC weights_data;
#if SUB_GROUP_SIZE == 8
            weights_data = AS_TYPE(FILTER_PACKED_TYPE_8, BLOCK_READ_8(weights + filter_idx));
#else
            weights_data.lo = AS_TYPE(FILTER_PACKED_TYPE_8, BLOCK_READ_8(weights + filter_idx));
            weights_data.hi = AS_TYPE(FILTER_PACKED_TYPE_8, BLOCK_READ_8(weights + filter_idx + SUB_GROUP_SIZE * 32));
#endif // SUB_GROUP_SIZE == 8

            dotProd = MMAD(activations, weights_data, dotProd);
        }
#endif  // HAS_FEATURE_LEFTOVERS

    if (OUT_FEATURE_NUM % SUB_GROUP_SIZE != 0 && feature >= OUT_FEATURE_NUM)
        return;

#if BIAS_TERM
#if BIAS_PER_OUTPUT
    const uint bias_index = GET_DATA_INDEX(BIAS, batch, feature, 0, 0);
#elif BIAS_PER_OFM
    const uint bias_index = feature;
#endif // BIAS_PER_OUTPUT

    float dequantized = (float)dotProd + biases[bias_index];
#else
    float dequantized = (float)dotProd;
#endif // BIAS_TERM

#if IS_3D
    const uint out_idx = OUTPUT_GET_INDEX(batch, skip_f, feature, 0);
#else
    const uint out_idx = OUTPUT_GET_INDEX(batch, feature, 0, 0);
#endif

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE res = FUSED_OPS_RESULT;

    output[out_idx] = res;
#else
    output[out_idx] = TO_OUTPUT_TYPE(dequantized);
#endif // HAS_FUSED_OPS

#if SLM_DIV_FACTOR > 1
    }
#endif
}

#undef INPUT_PACKED_TYPE_8
#undef FILTER_PACKED_TYPE_8
#undef INPUT_PACKED_TYPE_VEC
#undef FILTER_PACKED_TYPE_VEC

#undef BLOCK_READ
#undef BLOCK_READ_8

#undef MMAD
