// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/include_all.cl"
#include "include/common.cl"
#include "include/data_types.cl"


#define OUTPUT_TYPE_BLOCK               MAKE_VECTOR_TYPE(OUTPUT_TYPE, VEC_SIZE)
#define TO_TYPE(type, val)              CAT(convert_, type)(val)
#define TO_TYPE_SAT(type, val)          CAT(CAT(convert_, type), _sat)(val)

#if ELTWISE_BROADCAST
    #define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX_SAFE)(idx_order)
#else
    #define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX)(idx_order)
#endif

KERNEL(eltwise_b_fs_yx_fsv4)(INPUTS_DECLS
                              __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
, FUSED_OPS_DECLS
#endif
)
{
    const uint y = (uint)get_global_id(0) / OUTPUT_SIZE_X;
    const uint x = (uint)get_global_id(0) % OUTPUT_SIZE_X;
    const uint f_block = get_group_id(1);
    const uint b = get_global_id(2);

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
    if ((f_block*VEC_SIZE + VEC_SIZE) > OUTPUT_FEATURE_NUM) {
        for (uint fp = OUTPUT_FEATURE_NUM % VEC_SIZE; fp < VEC_SIZE; fp++) {
            out[fp] = OUTPUT_VAL_ZERO;
        }
    }
#endif

    vstore4(out, 0, &output[OUTPUT_GET_INDEX(b, (f_block*VEC_SIZE), y, x)]);
}

#undef OUTPUT_TYPE_BLOCK
#undef TO_TYPE
#undef TO_TYPE_SAT