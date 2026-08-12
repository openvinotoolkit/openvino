// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(eltwise_gpu_vload8)(INPUTS_DECLS
                           __global OUTPUT_TYPE* output)
{
    const uint global_id = get_global_id(0);

    VLOAD_DECLS

#if ACTIVATION_IN_ACCUMULATOR_TYPE
    MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 8) res;
#else
    MAKE_VECTOR_TYPE(OUTPUT_TYPE, 8) res;
#endif

    DO_ELTWISE

#if ACTIVATION_IN_ACCUMULATOR_TYPE
    MAKE_VECTOR_TYPE(OUTPUT_TYPE, 8) out = TO_OUTPUT_VECTOR_TYPE(ACTIVATION(res, ACTIVATION_PARAMS), 8);
#else
    res = ACTIVATION(res, ACTIVATION_PARAMS);
    MAKE_VECTOR_TYPE(OUTPUT_TYPE, 8) out = res;
#endif

    vstore8(out, global_id, output);

}
