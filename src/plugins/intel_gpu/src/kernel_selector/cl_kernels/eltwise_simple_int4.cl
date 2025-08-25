// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(eltwise_simple_int4)(INPUTS_DECLS
                           __global OUTPUT_TYPE* output)
{
    const uint global_id = get_global_id(2);
    const uint int4_id = global_id / 2;

    INPUT1_TYPE in0;

    if (global_id % 2 == 0) {
       in0 =  TO_INPUT1_TYPE(input0[int4_id] & 0xF);
    } else {
       in0 =  TO_INPUT1_TYPE(input0[int4_id] >> 4 & 0xF);
    }
    INPUT1_TYPE in1 = input1[global_id];

    output[global_id] = in0 * in1;

    //printf("id=%d, in eltwise_simple_int4, in0=%f, in1=%f value=%f\n", get_global_id(2), in0, in1, in0 * in1);
    /*
    VLOAD_DECLS

    MAKE_VECTOR_TYPE(OUTPUT_TYPE, 8) res;

    DO_ELTWISE

    res = ACTIVATION(res, ACTIVATION_PARAMS);

    vstore8(res, global_id, output);
    */

}
