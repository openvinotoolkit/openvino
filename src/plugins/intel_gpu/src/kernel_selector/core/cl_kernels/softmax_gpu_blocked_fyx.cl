// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"


KERNEL(softmax)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
    ACCUMULATOR_TYPE max_value = UNIT_VAL_MIN;
    ACCUMULATOR_TYPE data[CLASS_NUM];
    uint cls = 0;

    const uint b = get_global_id(0);

    for (uint f=0; f < INPUT0_FEATURE_NUM; ++f) {
#if INPUT0_DIMS == 5
        for (uint z=0; z < INPUT0_SIZE_Z; ++z) {
#endif
            for (uint y=0; y < INPUT0_SIZE_Y; ++y) {
                for (uint x=0; x < INPUT0_SIZE_X; ++x) {
#if INPUT0_DIMS == 5
                    const uint index = INPUT0_GET_INDEX(b, f, z, y, x);
#else
                    const uint index = INPUT0_GET_INDEX(b, f, y, x);
#endif

                    ACCUMULATOR_TYPE in = input[index];
                    max_value = max(max_value, in);
                    data[cls++] = in;
                }
            }
#if INPUT0_DIMS == 5
        }
#endif
    }

    ACCUMULATOR_TYPE denominator = 0.0;
    for (cls = 0; cls < CLASS_NUM; ++cls) {
        data[cls] = native_exp(data[cls] - max_value);
        denominator += data[cls];
    }

    cls = 0;

    for (uint f=0; f < INPUT0_FEATURE_NUM; ++f) {
#if INPUT0_DIMS == 5
        for (uint z=0; z < INPUT0_SIZE_Z; ++z) {
#endif
            for (uint y=0; y < INPUT0_SIZE_Y; ++y) {
                for (uint x=0; x < INPUT0_SIZE_X; ++x) {
                    const ACCUMULATOR_TYPE res = data[cls++] / denominator;

#if INPUT0_DIMS == 5
                    const uint output_idx = OUTPUT_GET_INDEX(b, f, z, y, x);
#else
                    const uint output_idx = OUTPUT_GET_INDEX(b, f, y, x);
#endif

#if HAS_FUSED_OPS
                    FUSED_OPS;
                    output[output_idx] = FUSED_OPS_RESULT;
#else
                    output[output_idx] = ACTIVATION(res, ACTIVATION_PARAMS);
#endif
                }
            }
#if INPUT0_DIMS == 5
        }
#endif
    }
}
