// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// TODO: Use GET_INDEX functions
// TODO: Write algorithm explanation
// TODO: Handle sorted
// TODO: Handle axis

inline uint FUNC(valToThreadIdx)(const INPUT0_TYPE val) {
    uint input_as_uint = fabs(val);
    // Add LOCAL_SIZE to be sure that modulo is different for all threads
    input_as_uint += LOCAL_SIZE;
    return input_as_uint % LOCAL_SIZE;
}

inline uint FUNC(getUniqueOffset)(const __local uint* unique_length, const uint local_id) {
    uint offset = 0;
    for (uint i = local_id + LOCAL_SIZE; i > LOCAL_SIZE; --i) {
        offset += unique_length[i - LOCAL_SIZE - 1];
    }
    return offset;
}

KERNEL(unique_ref)
(const __global INPUT0_TYPE* input,
 __global OUTPUT_TYPE* output,
 __global OUTPUT1_TYPE* output1,
 __global OUTPUT2_TYPE* output2,
 __global OUTPUT3_TYPE* output3) {
    const uint local_id = get_local_id(0);

    INPUT0_TYPE unique_arr[INPUT0_LENGTH];
    __local uint unique_length[LOCAL_SIZE];

    for (uint i = 0; i < INPUT0_LENGTH; ++i) {
        uint thread_idx = FUNC_CALL(valToThreadIdx)(input[i]);
        if (thread_idx == local_id) {
            bool unique = true;
            for (uint j = 0; j < unique_length[local_id]; ++j) {
                if (input[i] == unique_arr[j]) {
                    unique = false;
                }
            }
            if (unique) {
                unique_arr[unique_length[local_id]++] = input[i];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint offset = FUNC_CALL(getUniqueOffset)(unique_length, local_id);
    for (uint i = 0; i < unique_length[local_id]; ++i) {
        output[offset + i] = unique_arr[i];
    }
}
