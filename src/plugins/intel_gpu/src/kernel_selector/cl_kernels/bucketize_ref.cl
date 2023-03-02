// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/algorithm.cl"

#define GET_INDEX(prefix, ORDER) CAT(prefix, _GET_INDEX)(ORDER)

#if INPUT1_DIMS == 4
#    define BUCKET_GET_INDEX(local) INPUT1_GET_INDEX(local, 0, 0, 0)
#elif INPUT1_DIMS == 5
#    define BUCKET_GET_INDEX(local) INPUT1_GET_INDEX(local, 0, 0, 0, 0)
#elif INPUT1_DIMS == 6
#    define BUCKET_GET_INDEX(local) INPUT1_GET_INDEX(local, 0, 0, 0, 0, 0)
#endif

#ifdef WITH_RIGHT_BOUND
DECLARE_LOWER_BOUND(search, __global INPUT1_TYPE, INPUT0_TYPE, BUCKET_GET_INDEX)
#else
DECLARE_UPPER_BOUND(search, __global INPUT1_TYPE, INPUT0_TYPE, BUCKET_GET_INDEX)
#endif

KERNEL(bucketize_ref)(const __global INPUT0_TYPE* input, const __global INPUT1_TYPE* buckets, __global OUTPUT_TYPE* output) {
    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);

#if OUTPUT_DIMS == 4
#    define ORDER b, f, y, x
    const uint x = dim0;
    const uint y = dim1;
#elif OUTPUT_DIMS == 5
#    define ORDER b, f, z, y, x
    const uint x = dim0 % OUTPUT_SIZE_X;
    const uint y = dim0 / OUTPUT_SIZE_X;
    const uint z = dim1;
#elif OUTPUT_DIMS == 6
#    define ORDER b, f, w, z, y, x
    const uint x = dim0 % OUTPUT_SIZE_X;
    const uint y = dim0 / OUTPUT_SIZE_X;
    const uint z = dim1 % OUTPUT_SIZE_Z;
    const uint w = dim1 / OUTPUT_SIZE_Z;
#endif
    const uint f = dim2 % OUTPUT_FEATURE_NUM;
    const uint b = dim2 / OUTPUT_FEATURE_NUM;

    const uint index = GET_INDEX(INPUT0, ORDER);
    const uint bound_index = FUNC_CALL(search)(buckets, 0, INPUT1_LENGTH, input[index]);

    const uint out_index = GET_INDEX(OUTPUT, ORDER);
    output[out_index] = bound_index;
}

#undef GET_INDEX
#undef ORDER
#undef BUCKET_GET_INDEX