// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

// Gather accepts negative indices. Keep all memory accesses inside the bank
// even when a caller supplies an index outside the ScatterNDUpdate domain.
inline int FUNC(row_index)(long i, int size) {
    return (int)clamp(i < 0 ? i + size : i, (long)0, (long)(size - 1));
}

inline bool FUNC(reset_source)(const __global INPUT5_TYPE* resets, int count, int row, int slots) {
    for (int i = 0; i < count; ++i) if (FUNC(row_index)(resets[i],slots) == row) return true;
    return false;
}

KERNEL(recurrent_conv_cache_replay)(OPTIONAL_SHAPE_INFO_ARG
    __global INPUT0_TYPE* cache, const __global INPUT1_TYPE* row_map,
    const __global INPUT2_TYPE* input, const __global INPUT3_TYPE* weights,
    const __global INPUT4_TYPE* control, const __global INPUT5_TYPE* reset_rows,
    __global OUTPUT_TYPE* output, int B, int T, int N, int J, int P,
    __global float* staged) {
    const int C = INPUT2_SIZE_X;
    const size_t F = (size_t)(HISTORY + P) * C;
    row_map += INPUT1_OFFSET;
    control += INPUT4_OFFSET;
    reset_rows += INPUT5_OFFSET;
#if REPLAY_STAGE == 0
    const size_t index = get_global_id(0);
    if (index >= (size_t)N * F) return;
    const int mapping = index / F;
    const size_t feature = index % F;
    const int source = FUNC(row_index)(row_map[2 * mapping + 1],INPUT0_SIZE_Y);
    staged[index] = FUNC(reset_source)(reset_rows, J, source, INPUT0_SIZE_Y) ? 0.0f : convert_float(cache[INPUT0_GET_INDEX(0, 0, source, feature)]);
#else
    const size_t group = get_group_id(0), lane = get_local_id(0), local_size = get_local_size(0);
    const size_t output_count = (size_t)B * T * C;
    const size_t output_groups = (output_count + local_size - 1) / local_size;
    const size_t history_count = (size_t)B * HISTORY * C;
    const size_t history_groups = (history_count + local_size - 1) / local_size;
    const bool keep = control[B] != 0;
    if (group < output_groups) {
        const size_t index = group * local_size + lane;
        if (index >= output_count) return;
        const int c = index % C, t = (index / C) % T, b = index / ((size_t)T * C);
        const int replay = (int)clamp((long)control[b],-(long)(HISTORY+P),(long)P), destination = FUNC(row_index)(row_map[2 * b],INPUT0_SIZE_Y);
        float sum = 0.0f;
        for (int j = 0; j <= HISTORY; ++j) {
            const int column = t + j;
            const float value = column < HISTORY ? staged[b * F + FUNC(row_index)(replay + column,HISTORY + P) * C + c]
                                                  : convert_float(input[INPUT2_GET_INDEX(0, b, column - HISTORY, c)]);
            sum = fma(value, convert_float(weights[INPUT3_GET_INDEX(0, 0, c, j)]), sum);
        }
        output[OUTPUT_GET_INDEX(0, b, t, c)] = TO_OUTPUT_TYPE(sum);
        if (keep && t < P)
            cache[INPUT0_GET_INDEX(0, 0, destination, (HISTORY + t) * C + c)] =
                convert_float(input[INPUT2_GET_INDEX(0, b, t, c)]);
        return;
    }
    if (group < output_groups + history_groups) {
        const size_t index = (group - output_groups) * local_size + lane;
        if (index >= history_count) return;
        const int c = index % C, column = (index / C) % HISTORY, b = index / (HISTORY * C);
        const int source_column = keep ? column : T + column;
        const float value = source_column < HISTORY ? staged[b * F + FUNC(row_index)(clamp((long)control[b],-(long)(HISTORY+P),(long)P) + source_column,HISTORY + P) * C + c]
                                                    : convert_float(input[INPUT2_GET_INDEX(0, b, source_column - HISTORY, c)]);
        cache[INPUT0_GET_INDEX(0, 0, FUNC(row_index)(row_map[2 * b],INPUT0_SIZE_Y), column * C + c)] = value;
        return;
    }
    const size_t tiles = (F + local_size - 1) / local_size;
    const size_t remaining = group - output_groups - history_groups;
    const int work_row = remaining / tiles;
    const size_t feature = (remaining % tiles) * local_size + lane;
    if (feature >= F) return;
    if (work_row < N - B) {
        const int mapping = B + work_row, destination = FUNC(row_index)(row_map[2 * mapping],INPUT0_SIZE_Y);
        cache[INPUT0_GET_INDEX(0, 0, destination, feature)] = staged[mapping * F + feature];
    } else {
        const int reset_index = work_row - (N - B);
        if (reset_index >= J) return;
        const int destination = FUNC(row_index)(reset_rows[reset_index],INPUT0_SIZE_Y);
        for (int i = 0; i < N; ++i) {
            if (FUNC(row_index)(row_map[2 * i],INPUT0_SIZE_Y) != destination) continue;
            if (i >= B || feature < (size_t)(HISTORY + (keep ? T : 0)) * C) return;
            break;
        }
        cache[INPUT0_GET_INDEX(0, 0, destination, feature)] = 0.0f;
    }
#endif
}
