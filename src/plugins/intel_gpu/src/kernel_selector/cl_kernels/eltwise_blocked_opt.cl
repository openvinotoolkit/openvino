// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define OUTPUT_TYPE_BLOCK               MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE)
#define TO_TYPE(type, val)              CAT(convert_, type)(val)

#if ELTWISE_BROADCAST
    #define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX_SAFE)(idx_order)
#else
    #define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX)(idx_order)
#endif

KERNEL(eltwise_blocked_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    INPUTS_DECLS
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
    const uint xy = zyx % (uint)OUTPUT_SIZE_XY;
    const uint bf = zyx / (uint)OUTPUT_SIZE_XY;

    const uint x = xy % OUTPUT_SIZE_X;
    const uint y = xy / OUTPUT_SIZE_X;

    const uint outer_f = bf % (uint)OUT_F_BLOCK;
    const uint outer_b = bf / (uint)OUT_F_BLOCK;
#endif

    // Calculate batch and feature index for GET_INDEX_format(b, f_block, z, y, x)
    const uint b = inner_b + outer_b * INNER_BATCH_SIZE;
    const uint f_block = (inner_f + outer_f * F_BLOCK_COUNT);

    if (b >= OUTPUT_BATCH_NUM) {
        return;
    }

#if PADDED_OUTPUT
    if ((OUTPUT_PAD_BEFORE_FEATURE_NUM + f_block * VEC_SIZE) >=
        ALIGN(OUTPUT_PAD_BEFORE_FEATURE_NUM + OUTPUT_FEATURE_NUM + OUTPUT_PAD_AFTER_FEATURE_NUM, INNER_FEATURE_BLOCK_SIZE)) {
        return;
    }
#endif

    OUTPUT_TYPE_BLOCK out;
    if ((f_block * VEC_SIZE) >= OUTPUT_FEATURE_NUM) {
        out = (OUTPUT_TYPE_BLOCK)(TO_OUTPUT_TYPE(OUTPUT_VAL_ZERO));
    } else {
        MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, VEC_SIZE) res;
        DO_ELTWISE

#if HAS_FUSED_OPS
        FUSED_OPS;
        out = TO_TYPE(OUTPUT_TYPE_BLOCK, FUSED_OPS_RESULT);
#else
        out = TO_OUTPUT_VECTOR_TYPE(ACTIVATION_TYPED(TO_TYPE(MAKE_VECTOR_TYPE(OUTPUT_COMPUTE_TYPE, VEC_SIZE), res), ACTIVATION_PARAMS_TYPED), VEC_SIZE);
#endif

#ifdef LEFTOVERS
        // Overwrite unused feature lanes in the partial vector block with zero
        if ((f_block * VEC_SIZE + VEC_SIZE) > OUTPUT_FEATURE_NUM) {
            const uint valid_lanes = OUTPUT_FEATURE_NUM - f_block * VEC_SIZE;
            unroll_for (uint fp = 0; fp < VEC_SIZE; fp++) {
                if (fp >= valid_lanes) {
                    out[fp] = TO_OUTPUT_TYPE(OUTPUT_VAL_ZERO);
                }
            }
        }
#endif
    }

#if PADDED_OUTPUT
#if OUTPUT_DIMS == 5
    VSTORE_N(out, 0, &output[OUTPUT_GET_INDEX_RAW(b, (f_block * VEC_SIZE), z, y, x)]);
#else
    VSTORE_N(out, 0, &output[OUTPUT_GET_INDEX_RAW(b, (f_block * VEC_SIZE), y, x)]);
#endif
#else
    VSTORE_N(out, global_id, output);
#endif
}

#undef OUTPUT_TYPE_BLOCK
#undef TO_TYPE
