// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

KERNEL(bucketize_ref)(const __global INPUT0_TYPE* data,
                      const __global INPUT1_TYPE* buckets,
                      __global OUTPUT_TYPE* output)
{
    const uint batch = get_global_id(0);
    const uint feature = get_global_id(1);

#if OUTPUT_DIMS == 5
    const uint z = ((uint)get_global_id(2) / OUTPUT_SIZE_X) / OUTPUT_SIZE_Y;
    const uint y = ((uint)get_global_id(2) / OUTPUT_SIZE_X) % OUTPUT_SIZE_Y;
    const uint x = (uint)get_global_id(2) % OUTPUT_SIZE_X;
#else
    const uint z = 0;
    const uint y = (uint)get_global_id(2) / OUTPUT_SIZE_X;
    const uint x = (uint)get_global_id(2) % OUTPUT_SIZE_X;
#endif

    const uint input_index = INPUT0_GET_INDEX(batch, feature, z, y, x);
    const uint output_index = OUTPUT_GET_INDEX(batch, feature, z, y, x);
    for (int i = 0; i <= BUCKETS_SHAPE; i++) {
        if (WITH_RIGHT_BOUND) {
            if (data[input_index] <= buckets[i]) {
                output[output_index] = i;
                continue;
            }
            if (i == BUCKETS_SHAPE - 1) {
                output[output_index] = i + 1;
            }
        } else {
            if (data[input_index] < buckets[i]) {
                output[output_index] = i;
                continue;
            }
            if (i == BUCKETS_SHAPE - 1) {
                output[output_index] = i + 1;
            }
        }
    }
}
