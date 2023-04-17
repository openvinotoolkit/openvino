// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if FLATTENED
#    define LENGTH INPUT0_LENGTH
#else
#    define LENGTH AXIS_LENGTH
#endif

inline void FUNC(swap_out_unique_elements)(__global OUTPUT1_TYPE* a, __global OUTPUT1_TYPE* b) {
    const OUTPUT1_TYPE temp = *a;
    *a = *b;
    *b = temp;
}

inline void FUNC(swap_out_indices)(__global OUTPUT2_TYPE* a, __global OUTPUT2_TYPE* b) {
    const OUTPUT2_TYPE temp = *a;
    *a = *b;
    *b = temp;
}

#if !FLATTENED
inline bool FUNC(compare_slices_ascending)(OPTIONAL_SHAPE_INFO_ARG
                                           const __global OUTPUT1_TYPE* out_unique_elements,
                                           uint lhs,
                                           uint rhs) {
    ITERATE(
        if (out_unique_elements[GET_INDEX(OUTPUT1, lhs)] > out_unique_elements[GET_INDEX(OUTPUT1, rhs)]) {
            return true;
        } else if (out_unique_elements[GET_INDEX(OUTPUT1, lhs)] < out_unique_elements[GET_INDEX(OUTPUT1, rhs)]) {
            return false;
        } else { continue; })
    return false;
}

inline void FUNC(swap_slices)(OPTIONAL_SHAPE_INFO_ARG
                              __global OUTPUT1_TYPE* out_unique_elements,
                              uint lhs,
                              uint rhs) {
    ITERATE(FUNC_CALL(swap_out_unique_elements)(&out_unique_elements[GET_INDEX(OUTPUT1, lhs)],
                                                &out_unique_elements[GET_INDEX(OUTPUT1, rhs)]);)
}

inline bool FUNC(slices_are_equal)(OPTIONAL_SHAPE_INFO_ARG
                                   const __global OUTPUT1_TYPE* out_unique_elements,
                                   uint lhs,
                                   uint rhs) {
    ITERATE(if (out_unique_elements[GET_INDEX(OUTPUT1, lhs)] != out_unique_elements[GET_INDEX(OUTPUT1, rhs)]) {
        return false;
    })
    return true;
}

inline void FUNC(assign_slice)(OPTIONAL_SHAPE_INFO_ARG
                               __global OUTPUT1_TYPE* out_unique_elements,
                               uint lhs,
                               uint rhs) {
    ITERATE(out_unique_elements[GET_INDEX(OUTPUT1, lhs)] = out_unique_elements[GET_INDEX(OUTPUT1, rhs)];)
}

// We have almost the same versions of slices_are_equal and assign_slice functions, but here we use INPUT0 for GET_INDEX
inline bool FUNC(slices_are_equal_in)(OPTIONAL_SHAPE_INFO_ARG
                                      const __global OUTPUT1_TYPE* out_unique_elements,
                                      uint lhs,
                                      const __global INPUT0_TYPE* input,
                                      uint rhs) {
    ITERATE(if (out_unique_elements[GET_INDEX(OUTPUT1, lhs)] != input[GET_INDEX(INPUT0, rhs)]) { return false; })
    return true;
}

inline void FUNC(assign_slice_in)(OPTIONAL_SHAPE_INFO_ARG
                                  __global OUTPUT1_TYPE* out_unique_elements,
                                  uint lhs,
                                  const __global INPUT0_TYPE* input,
                                  uint rhs) {
    ITERATE(out_unique_elements[GET_INDEX(OUTPUT1, lhs)] = input[GET_INDEX(INPUT0, rhs)];)
}
#endif

// We use bubble sort here, because we need stable sort
// TODO: Change to better stable sort algorithm
inline void FUNC(bubbleSort)(OPTIONAL_SHAPE_INFO_ARG
                             __global OUTPUT1_TYPE* out_unique_elements,
                             __global OUTPUT2_TYPE* out_indices,
                             int l,
                             int h) {
    for (int i = 0; i < h - l; ++i) {
        bool swapped = false;
        for (int j = l; j < h - i; ++j) {
#if FLATTENED
            int j1 = j + 1;
            if ((out_unique_elements[GET_INDEX(OUTPUT1, j)] > out_unique_elements[GET_INDEX(OUTPUT1, j1)])) {
                FUNC_CALL(swap_out_unique_elements)
                (&out_unique_elements[GET_INDEX(OUTPUT1, j)], &out_unique_elements[GET_INDEX(OUTPUT1, j1)]);
#else
            if (FUNC_CALL(compare_slices_ascending)(OPTIONAL_SHAPE_INFO_TENSOR out_unique_elements, j, j + 1)) {
                FUNC_CALL(swap_slices)(OPTIONAL_SHAPE_INFO_TENSOR out_unique_elements, j, j + 1);
#endif
                FUNC_CALL(swap_out_indices)(&out_indices[j], &out_indices[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) {
            break;
        }
    }
}

// Works as std::unique, only on already sorted data
inline uint FUNC(deduplicate)(OPTIONAL_SHAPE_INFO_ARG
                              __global OUTPUT1_TYPE* out_unique_elements,
                              __global OUTPUT2_TYPE* out_indices,
                              __global OUTPUT3_TYPE* out_rev_indices,
                              __global OUTPUT4_TYPE* out_counts,
                              uint first,
                              const uint last) {
    if (first == last) {
        return last;
    }
    uint dest = first;
    out_rev_indices[out_indices[first]] = dest;
    ++out_counts[dest];
    while (++first != last) {
#if FLATTENED
        if (out_unique_elements[GET_INDEX(OUTPUT1, dest)] != out_unique_elements[GET_INDEX(OUTPUT1, first)]) {
            ++dest;
            out_unique_elements[GET_INDEX(OUTPUT1, dest)] = out_unique_elements[GET_INDEX(OUTPUT1, first)];
#else
        if (!FUNC_CALL(slices_are_equal)(OPTIONAL_SHAPE_INFO_TENSOR out_unique_elements, dest, first)) {
            FUNC_CALL(assign_slice)(OPTIONAL_SHAPE_INFO_TENSOR out_unique_elements, ++dest, first);
#endif
            out_indices[dest] = out_indices[first];
        }
        out_rev_indices[out_indices[first]] = dest;
        ++out_counts[dest];
    }
    return ++dest;
}

// Works on unsorted data, but has worse complexity
inline uint FUNC(unique)(OPTIONAL_SHAPE_INFO_ARG
                         const __global INPUT0_TYPE* input,
                         __global OUTPUT1_TYPE* out_unique_elements,
                         __global OUTPUT2_TYPE* out_indices,
                         __global OUTPUT3_TYPE* out_rev_indices,
                         __global OUTPUT4_TYPE* out_counts,
                         uint first,
                         const uint last) {
    uint unique_length = 0;
    for (; first != last; ++first) {
        bool unique = true;
        for (uint unique_idx = 0; unique_idx < unique_length; ++unique_idx) {
#if FLATTENED
            if (out_unique_elements[GET_INDEX(OUTPUT1, unique_idx)] == input[GET_INDEX(INPUT0, first)]) {
#else
            if (FUNC_CALL(slices_are_equal_in)(OPTIONAL_SHAPE_INFO_TENSOR out_unique_elements, unique_idx, input, first)) {
#endif
                unique = false;
                out_rev_indices[first] = unique_idx;
                ++out_counts[unique_idx];
                break;
            }
        }
        if (unique) {
#if FLATTENED
            out_unique_elements[GET_INDEX(OUTPUT1, unique_length)] = input[GET_INDEX(INPUT0, first)];
#else
            FUNC_CALL(assign_slice_in)(OPTIONAL_SHAPE_INFO_TENSOR out_unique_elements, unique_length, input, first);
#endif
            out_indices[unique_length] = first;
            out_rev_indices[first] = unique_length;
            ++out_counts[unique_length];
            ++unique_length;
        }
    }
    return unique_length;
}

KERNEL(unique_ref)
(OPTIONAL_SHAPE_INFO_ARG
 const __global INPUT0_TYPE* input,
 __global OUTPUT_TYPE* out_total_count,
 __global OUTPUT1_TYPE* out_unique_elements,
 __global OUTPUT2_TYPE* out_indices,
 __global OUTPUT3_TYPE* out_rev_indices,
 __global OUTPUT4_TYPE* out_counts) {
#if SORTED
    // Copy input to output data and initialize out_indices
    for (uint i = 0; i < LENGTH; ++i) {
#    if FLATTENED
        out_unique_elements[GET_INDEX(OUTPUT1, i)] = input[GET_INDEX(INPUT0, i)];
#    else
        FUNC_CALL(assign_slice_in)(OPTIONAL_SHAPE_INFO_TENSOR out_unique_elements, i, input, i);
#    endif
        out_indices[i] = i;
    }
    // Sort out_unique_elements together with out_indices
    FUNC_CALL(bubbleSort)(OPTIONAL_SHAPE_INFO_TENSOR out_unique_elements, out_indices, 0, LENGTH - 1);
    // Run deduplicate algorithm
    const uint end = FUNC_CALL(deduplicate)(OPTIONAL_SHAPE_INFO_TENSOR out_unique_elements, out_indices, out_rev_indices, out_counts, 0, LENGTH);
#else
    // Run unique algorithm
    const uint end = FUNC_CALL(unique)(OPTIONAL_SHAPE_INFO_TENSOR input, out_unique_elements, out_indices, out_rev_indices, out_counts, 0, LENGTH);
#endif
    out_total_count[0] = end;
}

#undef LENGTH
