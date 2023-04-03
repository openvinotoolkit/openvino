// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// TODO: Handle axis

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

// We use bubble sort here, because we need stable sort
// TODO: Change to better stable sort algorithm
inline void FUNC(bubbleSort)(__global OUTPUT_TYPE* out_unique_elements,
                             __global OUTPUT1_TYPE* out_indices,
                             uint l,
                             uint h) {
    for (uint i = 0; i < h - l; ++i) {
        bool swapped = false;
        for (uint j = l; j < h - i; ++j) {
            if ((out_unique_elements[j] > out_unique_elements[j + 1])) {
                FUNC_CALL(swap_out_unique_elements)(&out_unique_elements[j], &out_unique_elements[j + 1]);
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
inline uint FUNC(deduplicate)(__global OUTPUT_TYPE* out_unique_elements,
                              __global OUTPUT1_TYPE* out_indices,
                              __global OUTPUT2_TYPE* out_rev_indices,
                              __global OUTPUT3_TYPE* out_counts,
                              uint first,
                              const uint last) {
    if (first == last) {
        return last;
    }
    uint dest = first;
    out_rev_indices[out_indices[first]] = dest;
    ++out_counts[dest];
    while (++first != last) {
        if (out_unique_elements[dest] != out_unique_elements[first]) {
            out_unique_elements[++dest] = out_unique_elements[first];
            out_indices[dest] = out_indices[first];
        }
        out_rev_indices[out_indices[first]] = dest;
        ++out_counts[dest];
    }
    return ++dest;
}

// Works on unsorted data, but has worse complexity
inline uint FUNC(unique)(const __global INPUT0_TYPE* input,
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
            if (input[first] == out_unique_elements[unique_idx]) {
                unique = false;
                out_rev_indices[first] = unique_idx;
                ++out_counts[unique_idx];
                break;
            }
        }
        if (unique) {
            out_unique_elements[unique_length] = input[first];
            out_indices[unique_length] = first;
            out_rev_indices[first] = unique_length;
            ++out_counts[unique_length];
            ++unique_length;
        }
    }
    return unique_length;
}

KERNEL(unique_ref)
(const __global INPUT0_TYPE* input,
 __global OUTPUT_TYPE* out_unique_elements,
 __global OUTPUT1_TYPE* out_indices,
 __global OUTPUT2_TYPE* out_rev_indices,
 __global OUTPUT3_TYPE* out_counts
#ifdef OUTPUT4_TYPE
 ,
 __global OUTPUT4_TYPE* out_total_count
#endif
) {
#if FLATTENED
#    if SORTED
    // Copy input to output data and initialize out_indices
    for (uint i = 0; i < INPUT0_LENGTH; ++i) {
        out_unique_elements[i] = input[i];
        out_indices[i] = i;
    }
    // Sort out_unique_elements together with out_indices
    FUNC_CALL(bubbleSort)(out_unique_elements, out_indices, 0, INPUT0_LENGTH - 1);
    // Run deduplicate algorithm
    const uint end =
        FUNC_CALL(deduplicate)(out_unique_elements, out_indices, out_rev_indices, out_counts, 0, INPUT0_LENGTH);
#    else
    // Run unique algorithm
    const uint end =
        FUNC_CALL(unique)(input, out_unique_elements, out_indices, out_rev_indices, out_counts, 0, INPUT0_LENGTH);
#    endif
#    ifdef OUTPUT4_TYPE
    out_total_count[0] = end;
#    endif
#endif
}
