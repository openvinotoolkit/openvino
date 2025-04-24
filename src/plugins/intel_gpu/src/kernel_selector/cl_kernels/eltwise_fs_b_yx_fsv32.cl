// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(eltwise_fs_b_yx_fsv32)(
    INPUTS_DECLS
    __global UNIT_TYPE* output)
{
    const uint global_id = get_global_id(0);

    VLOAD_DECLS

    MAKE_VECTOR_TYPE(UNIT_TYPE, 8) res;

    DO_ELTWISE

    res = ACTIVATION(res, ACTIVATION_PARAMS);

    vstore8(res, global_id, output);

}
