// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/algorithm.cl"

#ifdef WITH_RIGHT_BOUND
DECLARE_LOWER_BOUND(search, const __global INPUT1_TYPE, INPUT0_TYPE)
#else
DECLARE_UPPER_BOUND(search, const __global INPUT1_TYPE, INPUT0_TYPE)
#endif

#define GET_INDEX(prefix, ORDER) CAT(prefix, _GET_INDEX)(ORDER)

KERNEL(bucketize_ref)
(const __global INPUT0_TYPE* input, const __global INPUT1_TYPE* buckets, __global OUTPUT_TYPE* output) {
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

    const __global INPUT1_TYPE* bound = FUNC_CALL(search)(buckets, buckets + INPUT1_LENGTH, input[index]);

    output[index] = bound - buckets;
}

#undef GET_INDEX
#undef ORDER
