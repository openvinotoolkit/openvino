// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if FLATTENED
#    define LENGTH INPUT1_LENGTH
#else
#    define LENGTH AXIS_LENGTH
#endif

KERNEL(unique_reshape_ref)
(OPTIONAL_SHAPE_INFO_ARG
 const __global INPUT0_TYPE* in_total_count,
 const __global INPUT1_TYPE* in_unique_elements,
 const __global INPUT2_TYPE* in_indices,
 const __global INPUT3_TYPE* in_rev_indices,
 const __global INPUT4_TYPE* in_counts,
 __global OUTPUT_TYPE* out_unique_elements,
 __global OUTPUT1_TYPE* out_indices,
 __global OUTPUT2_TYPE* out_rev_indices,
 __global OUTPUT3_TYPE* out_counts) {
    // Copy all data to new shape
    for (uint i = 0; i < in_total_count[0]; ++i) {
#if FLATTENED
        out_unique_elements[i] = in_unique_elements[i];
#else
        ITERATE(out_unique_elements[GET_INDEX(OUTPUT, i)] = in_unique_elements[GET_INDEX(INPUT1, i)];)
#endif
        out_indices[i] = in_indices[i];
        out_rev_indices[i] = in_rev_indices[i];
        out_counts[i] = in_counts[i];
    }

    // rev_indices always has whole shape, need to additionally copy remained data
    for (uint i = in_total_count[0]; i < LENGTH; ++i) {
        out_rev_indices[i] = in_rev_indices[i];
    }
}

#undef LENGTH
