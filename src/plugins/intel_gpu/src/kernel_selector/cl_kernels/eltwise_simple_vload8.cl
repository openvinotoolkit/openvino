// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(eltwise_gpu_vload8)(INPUTS_DECLS
                           __global OUTPUT_TYPE* output)
{
    const uint global_id = get_global_id(0);

    VLOAD_DECLS

    MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 8) res;

    DO_ELTWISE

    MAKE_VECTOR_TYPE(OUTPUT_TYPE, 8) out = TO_OUTPUT_VECTOR_TYPE(ACTIVATION(res, ACTIVATION_PARAMS),8);

    vstore8(out, global_id, output);

}
