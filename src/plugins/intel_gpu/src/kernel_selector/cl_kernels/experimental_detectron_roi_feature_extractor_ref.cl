// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

inline int FUNC(get_pyramid_level_index)(uint level, uint c, uint y, uint x) {
    uint idx = 0;
    LEVELS_IDX_CALC_FUNCS;
    return idx;
}

inline int FUNC(get_pyramid_level_for_roi)(const __global INPUT0_TYPE* current_roi) {
    const INPUT0_TYPE canonical_scale = 224.0;
    const int canonical_level = 2;

    int result = NUM_PYRAMID_LEVELS;

    const INPUT0_TYPE x0 = current_roi[0];
    const INPUT0_TYPE y0 = current_roi[1];
    const INPUT0_TYPE x1 = current_roi[2];
    const INPUT0_TYPE y1 = current_roi[3];

    const INPUT0_TYPE area = (x1 - x0) * (y1 - y0);
    if (area > 0) {
        result = (int)round(canonical_level + log2(sqrt(area) / canonical_scale));
        result = max(0, min(result, NUM_PYRAMID_LEVELS - 1));
    }
    return result;
}

KERNEL(experimental_detectron_roi_feature_extractor_ref)(const __global INPUT0_TYPE* src_rois,
                                                         INPUT_LEVEL_PARAMS,
                                                         __global OUTPUT_TYPE* dst_data)
{
    const uint oxy = get_global_id(0);

    const uint x = oxy % POOLED_WIDTH;
    const uint y = oxy / POOLED_WIDTH;
    const uint c = get_global_id(1);
    const uint r = get_global_id(2);

    const __global INPUT0_TYPE* current_roi_ptr = &src_rois[r * INPUT0_BATCH_PITCH];

    const int level = FUNC_CALL(get_pyramid_level_for_roi)(current_roi_ptr);

    const __global INPUT1_TYPE* current_level_ptr = LEVEL_PTRS[level];

    INPUT0_TYPE offset = IS_ALIGNED ? TO_INPUT0_TYPE(0.5f) : TO_INPUT0_TYPE(0.0);

    INPUT0_TYPE spatial_scale = SPATIAL_SCALES[level];
    INPUT0_TYPE roi_start_w = current_roi_ptr[0] * spatial_scale - offset;
    INPUT0_TYPE roi_start_h = current_roi_ptr[1] * spatial_scale - offset;
    INPUT0_TYPE roi_end_w = current_roi_ptr[2] * spatial_scale - offset;
    INPUT0_TYPE roi_end_h = current_roi_ptr[3] * spatial_scale - offset;

    INPUT0_TYPE roi_width = max(roi_end_w - roi_start_w, TO_INPUT0_TYPE(1.));
    INPUT0_TYPE roi_height = max(roi_end_h - roi_start_h, TO_INPUT0_TYPE(1.));

    INPUT0_TYPE bin_width = roi_width / TO_INPUT0_TYPE(POOLED_WIDTH);
    INPUT0_TYPE bin_height = roi_height / TO_INPUT0_TYPE(POOLED_HEIGHT);

    const uint roi_bin_grid_w = (SAMPLING_RATIO > 0) ? SAMPLING_RATIO : (uint)ceil(roi_width / POOLED_WIDTH);
    const uint roi_bin_grid_h = (SAMPLING_RATIO > 0) ? SAMPLING_RATIO : (uint)ceil(roi_height / POOLED_HEIGHT);

    size_t level_sizes_arr[3*NUM_PYRAMID_LEVELS] = LEVEL_SIZES;
    const uint level_h = level_sizes_arr[3 * level];
    const uint level_w = level_sizes_arr[3 * level + 1];
    const uint level_offset = level_sizes_arr[3 * level + 2];

    INPUT0_TYPE output_val = 0.0;
    INPUT0_TYPE current_bin_start_h = roi_start_h + y * bin_height;
    INPUT0_TYPE current_bin_start_w = roi_start_w + x * bin_width;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
        INPUT0_TYPE yy = current_bin_start_h + TO_INPUT0_TYPE(iy + 0.5f) * bin_height / TO_INPUT0_TYPE(roi_bin_grid_h);
        if (yy < -1.0 || yy > level_h) {
            continue;
        }
        if (yy <= 0) {
            yy = 0.0f;
        }
        int y_low = (int)floor(yy);
        int y_high = 0;

        if (y_low >= level_h - 1) {
            y_high = y_low = level_h - 1;
            yy = TO_INPUT0_TYPE(y_low);
        } else {
            y_high = y_low + 1;
        }

        INPUT0_TYPE ly = yy - y_low;
        INPUT0_TYPE hy = TO_INPUT0_TYPE(1.0f) - ly;

        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
            INPUT0_TYPE xx = current_bin_start_w + TO_INPUT0_TYPE(ix + 0.5f) * bin_width / TO_INPUT0_TYPE(roi_bin_grid_w);
            if (xx < -1.0 || xx > level_w) {
                continue;
            }
            if (xx <= 0) {
                xx = 0.0f;
            }
            int x_low = (int)floor(xx);
            int x_high = 0;

            if (x_low >= level_w - 1) {
                x_high = x_low = level_w - 1;
                xx = TO_INPUT0_TYPE(x_low);
            } else {
                x_high = x_low + 1;
            }

            INPUT0_TYPE lx = xx - x_low;
            INPUT0_TYPE hx = TO_INPUT0_TYPE(1.0f) - lx;

            INPUT0_TYPE w1 = hy * hx;
            INPUT0_TYPE w2 = hy * lx;
            INPUT0_TYPE w3 = ly * hx;
            INPUT0_TYPE w4 = ly * lx;

            output_val += w1 * current_level_ptr[FUNC_CALL(get_pyramid_level_index)(level, c, y_low, x_low)] +
                          w2 * current_level_ptr[FUNC_CALL(get_pyramid_level_index)(level, c, y_low, x_high)] +
                          w3 * current_level_ptr[FUNC_CALL(get_pyramid_level_index)(level, c, y_high, x_low)] +
                          w4 * current_level_ptr[FUNC_CALL(get_pyramid_level_index)(level, c, y_high, x_high)];
        }
    }
    output_val /= TO_INPUT0_TYPE(roi_bin_grid_h * roi_bin_grid_w);
    const uint output_offset = OUTPUT_OFFSET + x * OUTPUT_X_PITCH + y * OUTPUT_Y_PITCH + c * OUTPUT_FEATURE_PITCH + r * OUTPUT_BATCH_PITCH;
    dst_data[output_offset] = output_val;
}
