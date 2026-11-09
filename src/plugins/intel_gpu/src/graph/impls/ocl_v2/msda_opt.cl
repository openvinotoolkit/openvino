// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Multi-scale deformable attention. Every work item computes one output
// element (b, q, m, c): the sum over levels and points of the attention weight
// times value bilinearly sampled at the point location, with zero padding and
// align_corners=false, which is the GridSample sampling the fusion replaces.
//
// The pixel coordinates and the bilinear weights are float, as in
// grid_sample_opt_bilinear_zeros.cl; the samples are accumulated in
// ACCUMULATOR_TYPE (f32) and converted to OUTPUT_TYPE on store. One work item
// per output channel keeps the corner loads of neighboring work items
// contiguous along D.
//
// SPATIAL_SIZE, NUM_QUERY, NUM_HEADS, EMBED_DIMS, NUM_LEVELS and NUM_POINT are
// JIT constants; the inputs are plain, so they are indexed linearly.

KERNEL(multi_scale_deformable_attn)(
    __global const INPUT0_TYPE* restrict data_value,         // (bs, num_keys, NUM_HEADS, EMBED_DIMS)
    __global const int* restrict data_spatial_shapes,        // (NUM_LEVELS, 2), (h, w) of every level
    __global const int* restrict data_level_start_index,     // (NUM_LEVELS), first key of every level
    __global const INPUT3_TYPE* restrict data_sampling_loc,  // (bs, num_queries, NUM_HEADS, NUM_LEVELS, NUM_POINT, 2), level-normalized (x, y)
    __global const INPUT4_TYPE* restrict data_attn_weight,   // (bs, num_queries, NUM_HEADS, NUM_LEVELS, NUM_POINT)
    __global OUTPUT_TYPE* restrict output) {                 // (bs, num_queries, NUM_HEADS * EMBED_DIMS)
    const int index = get_global_id(2);
    const int c = index % EMBED_DIMS;
    const int sampling_index = index / EMBED_DIMS;  // (b, q, m)
    const int m = sampling_index % NUM_HEADS;
    const int b = sampling_index / (NUM_HEADS * NUM_QUERY);

    const int key_stride = NUM_HEADS * EMBED_DIMS;
    __global const INPUT0_TYPE* batch_value = data_value + b * SPATIAL_SIZE * key_stride + m * EMBED_DIMS + c;
    int weight_index = sampling_index * NUM_LEVELS * NUM_POINT;

    ACCUMULATOR_TYPE acc = 0;
    for (int l = 0; l < NUM_LEVELS; ++l) {
        const int height = data_spatial_shapes[2 * l];
        const int width = data_spatial_shapes[2 * l + 1];
        __global const INPUT0_TYPE* level_value = batch_value + data_level_start_index[l] * key_stride;

        for (int p = 0; p < NUM_POINT; ++p, ++weight_index) {
            // GridSample unnormalizes the coordinate 2 * x - 1 to
            // ((2 * x - 1 + 1) * width - 1) / 2 = x * width - 0.5.
            const float x = (float)data_sampling_loc[2 * weight_index] * width - 0.5f;
            const float y = (float)data_sampling_loc[2 * weight_index + 1] * height - 0.5f;
            const ACCUMULATOR_TYPE attention = TO_ACCUMULATOR_TYPE(data_attn_weight[weight_index]);

            const int x0 = (int)floor(x);
            const int y0 = (int)floor(y);
            const float dx = x - x0;
            const float dy = y - y0;

            const bool x0_valid = x0 >= 0 && x0 < width;
            const bool x1_valid = x0 + 1 >= 0 && x0 + 1 < width;
            const bool y0_valid = y0 >= 0 && y0 < height;
            const bool y1_valid = y0 + 1 >= 0 && y0 + 1 < height;
            const int x0c = x0_valid ? x0 : 0;
            const int x1c = x1_valid ? x0 + 1 : 0;
            const int y0c = y0_valid ? y0 : 0;
            const int y1c = y1_valid ? y0 + 1 : 0;

            // The corners are loaded unconditionally from clamped, in-bounds
            // offsets and masked afterwards, which avoids divergent branches
            // around the loads (see LOAD_INPUT in grid_sample_opt_bilinear_zeros.cl).
            const INPUT0_TYPE v00_d = level_value[(y0c * width + x0c) * key_stride];
            const INPUT0_TYPE v01_d = level_value[(y0c * width + x1c) * key_stride];
            const INPUT0_TYPE v10_d = level_value[(y1c * width + x0c) * key_stride];
            const INPUT0_TYPE v11_d = level_value[(y1c * width + x1c) * key_stride];

            const ACCUMULATOR_TYPE v00 = (y0_valid && x0_valid) ? TO_ACCUMULATOR_TYPE(v00_d) * (1 - dx) : 0;
            const ACCUMULATOR_TYPE v01 = (y0_valid && x1_valid) ? TO_ACCUMULATOR_TYPE(v01_d) * dx : 0;
            const ACCUMULATOR_TYPE v10 = (y1_valid && x0_valid) ? TO_ACCUMULATOR_TYPE(v10_d) * (1 - dx) : 0;
            const ACCUMULATOR_TYPE v11 = (y1_valid && x1_valid) ? TO_ACCUMULATOR_TYPE(v11_d) * dx : 0;

            acc += attention * ((1 - dy) * (v00 + v01) + dy * (v10 + v11));
        }
    }
    output[index] = TO_OUTPUT_TYPE(acc);
}
