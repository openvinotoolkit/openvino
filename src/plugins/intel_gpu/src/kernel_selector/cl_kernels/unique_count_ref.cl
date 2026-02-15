// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef FLATTENED
#    define LENGTH TOTAL_DATA_SIZE
#else
#    define LENGTH AXIS_LENGTH
#endif

#ifndef FLATTENED
inline bool FUNC(slices_are_equal)(OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* out_unique_elements,
                                   uint lhs,
                                   const __global INPUT0_TYPE* input,
                                   uint rhs) {
    ITERATE(if (out_unique_elements[GET_INDEX(INPUT0, lhs)] != input[GET_INDEX(INPUT0, rhs)]) { return false; })
    return true;
}

inline void FUNC(assign_slice)(OPTIONAL_SHAPE_INFO_ARG __global INPUT0_TYPE* out_unique_elements,
                               uint lhs,
                               const __global INPUT0_TYPE* input,
                               uint rhs) {
    ITERATE(out_unique_elements[GET_INDEX(INPUT0, lhs)] = input[GET_INDEX(INPUT0, rhs)];)
}
#endif

// Works on unsorted data, but has worse complexity
inline uint FUNC(unique)(OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* input,
                         __global INPUT0_TYPE* out_unique_elements,
                         uint first,
                         const uint last) {
    uint unique_length = 0;
    for (; first != last; ++first) {
        bool unique = true;
        for (uint unique_idx = 0; unique_idx < unique_length; ++unique_idx) {
#ifdef FLATTENED
            if (out_unique_elements[unique_idx] == input[GET_INDEX(INPUT0, first)]) {
#else
            if (FUNC_CALL(slices_are_equal)(OPTIONAL_SHAPE_INFO_TENSOR out_unique_elements, unique_idx, input, first)) {
#endif
                unique = false;
                break;
            }
        }
        if (unique) {
#ifdef FLATTENED
            out_unique_elements[unique_length] = input[GET_INDEX(INPUT0, first)];
#else
            FUNC_CALL(assign_slice)(OPTIONAL_SHAPE_INFO_TENSOR out_unique_elements, unique_length, input, first);
#endif
            ++unique_length;
        }
    }
    return unique_length;
}

KERNEL(unique_count_ref)
(OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* input,
 __global OUTPUT_TYPE* out_total_count,
 __global INPUT0_TYPE* out_unique_elements) {
    out_total_count[0] = FUNC_CALL(unique)(OPTIONAL_SHAPE_INFO_TENSOR input, out_unique_elements, 0, LENGTH);
}

#undef LENGTH
