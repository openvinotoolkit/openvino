// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define OUTPUT_TYPE_BLOCK               MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE)
#define TO_TYPE(type, val)              CAT(convert_, type)(val)
#define TO_TYPE_SAT(type, val)          CAT(CAT(convert_, type), _sat)(val)

#if ELTWISE_BROADCAST
    #define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX_SAFE)(idx_order)
#else
    #define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX)(idx_order)
#endif

KERNEL(eltwise_blocked_opt)(INPUTS_DECLS
                              __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
, FUSED_OPS_DECLS
#endif
)
{
    const uint global_id = get_global_id(0);

    // For double blocked formats, calculate its size of inner blocks (both batch & feature) : INNER_BLOCKS_COUNT = INNER_BATCH_SIZE * F_BLOCK_COUNT
    const uint inner_block = global_id % INNER_BLOCKS_COUNT;
    // Calculate index of feature axis inner block which is divided by vector size
    uint inner_f = inner_block % F_BLOCK_COUNT;
    // Calculate index of batch axis inner block
    uint inner_b = inner_block / F_BLOCK_COUNT;

    const uint zyx = (uint)global_id / INNER_BLOCKS_COUNT;

#if OUTPUT_DIMS == 5
    const uint yx = zyx % (uint)OUTPUT_SIZE_XY;
    const uint bfz = zyx / (uint)OUTPUT_SIZE_XY;

    const uint x = yx % OUTPUT_SIZE_X;
    const uint y = yx / OUTPUT_SIZE_X;

    const uint z = bfz % (uint)OUTPUT_SIZE_Z;
    const uint bf = bfz / (uint)OUTPUT_SIZE_Z;

    const uint outer_f = bf % (uint)OUT_F_BLOCK;
    const uint outer_b = bf / (uint)OUT_F_BLOCK;
#else
    const uint z = 0;
    const uint x = zyx % OUTPUT_SIZE_X;
    const uint bfy = zyx / OUTPUT_SIZE_X;

    const uint y = bfy % OUTPUT_SIZE_Y;
    const uint bf = bfy / OUTPUT_SIZE_Y;

    const uint outer_f = bf % (uint)OUT_F_BLOCK;
    const uint outer_b = bf / (uint)OUT_F_BLOCK;
#endif

    // Calculate batch and feature index for GET_INDEX_format(b, f_block, z, y, x)
    const uint b = inner_b + outer_b * INNER_BATCH_SIZE;
    const uint f_block = (inner_f + outer_f * F_BLOCK_COUNT);

    // Feature axis of input tensor is smaller than inner block size : No need to calculate this block
    if (b > OUTPUT_BATCH_NUM || (f_block*VEC_SIZE + VEC_SIZE) > (OUT_F_BLOCK * FEATURE_BLOCK_SIZE)) {
        return;
    }

    // Fill padded memory with zeros for b_fs_yx_fsv format
    if ((f_block*VEC_SIZE) >= OUTPUT_FEATURE_NUM && FEATURE_BLOCK_SIZE != 1) {
        MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, VEC_SIZE) out = (MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, VEC_SIZE))(0);
        vstore8(out, global_id, output);
        return;
    }

    MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, VEC_SIZE) res;

    DO_ELTWISE

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE_BLOCK out = TO_TYPE(MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE), FUSED_OPS_RESULT);
#else
#if QUANTIZATION_TERM && !OUTPUT_IS_FP
    OUTPUT_TYPE_BLOCK out = ACTIVATION_TYPED(TO_TYPE_SAT(MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE), res), ACTIVATION_PARAMS_TYPED);
#else
    OUTPUT_TYPE_BLOCK out = ACTIVATION_TYPED(TO_TYPE(MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE), res), ACTIVATION_PARAMS_TYPED);
#endif
#endif

#ifdef LEFTOVERS
    // Overwrite
    if ((f_block*VEC_SIZE + VEC_SIZE) > OUTPUT_FEATURE_NUM) {
        for (uint fp = OUTPUT_FEATURE_NUM % VEC_SIZE; fp < VEC_SIZE; fp++) {
            out[fp] = OUTPUT_VAL_ZERO;
        }
    }
#endif

#if PADDED_OUTPUT
#if OUTPUT_DIMS == 5
    VSTORE_N(out, 0, &output[OUTPUT_GET_INDEX(b, (f_block*VEC_SIZE), z, y, x)]);
#else
    VSTORE_N(out, 0, &output[OUTPUT_GET_INDEX(b, (f_block*VEC_SIZE), y, x)]);
#endif
#else
    VSTORE_N(out, global_id, output);
#endif
}

#undef OUTPUT_TYPE_BLOCK
#undef TO_TYPE
#undef TO_TYPE_SAT
