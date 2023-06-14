// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef FLATTENED
#    define LENGTH TOTAL_DATA_SIZE
#else
#    define LENGTH AXIS_LENGTH
#endif

inline void FUNC(swap_out_unique_elements)(__global OUTPUT_TYPE* a, __global OUTPUT_TYPE* b) {
    const OUTPUT_TYPE temp = *a;
    *a = *b;
    *b = temp;
}

inline void FUNC(swap_out_indices)(__global OUTPUT1_TYPE* a, __global OUTPUT1_TYPE* b) {
    const OUTPUT1_TYPE temp = *a;
    *a = *b;
    *b = temp;
}

inline void FUNC(swap_out_counts)(__global OUTPUT3_TYPE* a, __global OUTPUT3_TYPE* b) {
    const OUTPUT3_TYPE temp = *a;
    *a = *b;
    *b = temp;
}

#ifndef FLATTENED
inline bool FUNC(compare_slices_ascending)(OPTIONAL_SHAPE_INFO_ARG const __global OUTPUT_TYPE* out_unique_elements,
                                           uint lhs,
                                           uint rhs) {
    ITERATE(
        if (out_unique_elements[GET_INDEX(OUTPUT, lhs)] > out_unique_elements[GET_INDEX(OUTPUT, rhs)]) {
            return true;
        } else if (out_unique_elements[GET_INDEX(OUTPUT, lhs)] < out_unique_elements[GET_INDEX(OUTPUT, rhs)]) {
            return false;
        } else { continue; })
    return false;
}

inline void FUNC(swap_slices)(OPTIONAL_SHAPE_INFO_ARG __global OUTPUT_TYPE* out_unique_elements, uint lhs, uint rhs) {
    ITERATE(FUNC_CALL(swap_out_unique_elements)(&out_unique_elements[GET_INDEX(OUTPUT, lhs)],
                                                &out_unique_elements[GET_INDEX(OUTPUT, rhs)]);)
}

inline bool FUNC(slices_are_equal)(OPTIONAL_SHAPE_INFO_ARG const __global OUTPUT_TYPE* out_unique_elements,
                                   uint lhs,
                                   const __global INPUT0_TYPE* input,
                                   uint rhs) {
    ITERATE(if (out_unique_elements[GET_INDEX(OUTPUT, lhs)] != input[GET_INDEX(INPUT0, rhs)]) { return false; })
    return true;
}

inline void FUNC(assign_slice)(OPTIONAL_SHAPE_INFO_ARG __global OUTPUT_TYPE* out_unique_elements,
                               uint lhs,
                               const __global INPUT0_TYPE* input,
                               uint rhs) {
    ITERATE(out_unique_elements[GET_INDEX(OUTPUT, lhs)] = input[GET_INDEX(INPUT0, rhs)];)
}
#endif

// We use bubble sort here, because we need stable sort
// TODO: Change to better stable sort algorithm
inline void FUNC(bubbleSort)(OPTIONAL_SHAPE_INFO_ARG __global OUTPUT_TYPE* out_unique_elements,
                             __global OUTPUT1_TYPE* out_indices,
                             __global OUTPUT3_TYPE* out_counts,
                             int l,
                             int h) {
    for (int i = 0; i < h - l; ++i) {
        bool swapped = false;
        for (int j = l; j < h - i; ++j) {
#ifdef FLATTENED
            int j1 = j + 1;
            if ((out_unique_elements[GET_INDEX(OUTPUT, j)] > out_unique_elements[GET_INDEX(OUTPUT, j1)])) {
                FUNC_CALL(swap_out_unique_elements)
                (&out_unique_elements[GET_INDEX(OUTPUT, j)], &out_unique_elements[GET_INDEX(OUTPUT, j1)]);
#else
            if (FUNC_CALL(compare_slices_ascending)(OPTIONAL_SHAPE_INFO_TENSOR out_unique_elements, j, j + 1)) {
                FUNC_CALL(swap_slices)(OPTIONAL_SHAPE_INFO_TENSOR out_unique_elements, j, j + 1);
#endif
                FUNC_CALL(swap_out_indices)(&out_indices[j], &out_indices[j + 1]);
                FUNC_CALL(swap_out_counts)(&out_counts[j], &out_counts[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) {
            break;
        }
    }
}

// Works on unsorted data, but has worse complexity
inline uint FUNC(unique)(OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* input,
                         __global OUTPUT_TYPE* out_unique_elements,
                         __global OUTPUT1_TYPE* out_indices,
                         __global OUTPUT2_TYPE* out_rev_indices,
                         __global OUTPUT3_TYPE* out_counts,
                         uint first,
                         const uint last) {
    uint unique_length = 0;
    for (; first != last; ++first) {
        bool unique = true;
        for (uint unique_idx = 0; unique_idx < unique_length; ++unique_idx) {
#ifdef FLATTENED
            if (out_unique_elements[GET_INDEX(OUTPUT, unique_idx)] == input[GET_INDEX(INPUT0, first)]) {
#else
            if (FUNC_CALL(slices_are_equal)(OPTIONAL_SHAPE_INFO_TENSOR out_unique_elements, unique_idx, input, first)) {
#endif
                unique = false;
                out_rev_indices[first] = unique_idx;
                ++out_counts[unique_idx];
                break;
            }
        }
        if (unique) {
#ifdef FLATTENED
            out_unique_elements[GET_INDEX(OUTPUT, unique_length)] = input[GET_INDEX(INPUT0, first)];
#else
            FUNC_CALL(assign_slice)(OPTIONAL_SHAPE_INFO_TENSOR out_unique_elements, unique_length, input, first);
#endif
            out_indices[unique_length] = first;
            out_rev_indices[first] = unique_length;
            ++out_counts[unique_length];
            ++unique_length;
        }
    }
    return unique_length;
}

inline uint FUNC(fill_out_rev_indices)(OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* input,
                                       __global OUTPUT_TYPE* out_unique_elements,
                                       __global OUTPUT2_TYPE* out_rev_indices,
                                       const uint end) {
    for (uint i = 0; i < LENGTH; ++i) {
        for (uint j = 0; j < end; ++j) {
#ifdef FLATTENED
            if (out_unique_elements[GET_INDEX(OUTPUT, j)] == input[GET_INDEX(INPUT0, i)]) {
#else
            if (FUNC_CALL(slices_are_equal)(OPTIONAL_SHAPE_INFO_TENSOR out_unique_elements, j, input, i)) {
#endif
                out_rev_indices[i] = j;
                break;
            }
        }
    }
}

KERNEL(unique_gather_ref)
(OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* input,
 const __global INPUT1_TYPE* out_total_count,
 __global OUTPUT_TYPE* out_unique_elements,
 __global OUTPUT1_TYPE* out_indices,
 __global OUTPUT2_TYPE* out_rev_indices,
 __global OUTPUT3_TYPE* out_counts) {
    // TODO: Think of better approach to initialize with zeros
    for (uint i = 0; i < LENGTH; ++i) {
        out_counts[i] = 0;
    }
    // Run unique algorithm
    const uint end = FUNC_CALL(unique)(OPTIONAL_SHAPE_INFO_TENSOR input,
                                       out_unique_elements,
                                       out_indices,
                                       out_rev_indices,
                                       out_counts,
                                       0,
                                       LENGTH);
#ifdef SORTED
    // Sort out data
    FUNC_CALL(bubbleSort)(OPTIONAL_SHAPE_INFO_TENSOR out_unique_elements, out_indices, out_counts, 0, end - 1);
    // After sorting all out_unique_elements will shuffle and out_rev_indices should change not only order, but their
    // values (indexes).
    // So, we need to fill them again...
    // Another approach would be to allocate whole separate buffer as input, sort whole dataset first and then run
    // deduplicate algorithm with correct filling of out_rev_indices.
    FUNC_CALL(fill_out_rev_indices)
    (OPTIONAL_SHAPE_INFO_TENSOR input, out_unique_elements, out_rev_indices, end);
#endif
}

#undef LENGTH
