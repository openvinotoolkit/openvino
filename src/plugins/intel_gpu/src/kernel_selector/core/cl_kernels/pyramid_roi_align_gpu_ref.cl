// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

#define PYRAMID_LEVELS             4

struct Parameters
{
    int size_y, size_x, f_pitch, x_pitch, y_pitch, offset;
};

__constant struct Parameters parameters [PYRAMID_LEVELS] =
        {
            { INPUT1_SIZE_Y, INPUT1_SIZE_X, INPUT1_FEATURE_PITCH, INPUT1_X_PITCH, INPUT1_Y_PITCH, INPUT1_OFFSET },
            { INPUT2_SIZE_Y, INPUT2_SIZE_X, INPUT2_FEATURE_PITCH, INPUT2_X_PITCH, INPUT2_Y_PITCH, INPUT2_OFFSET },
            { INPUT3_SIZE_Y, INPUT3_SIZE_X, INPUT3_FEATURE_PITCH, INPUT3_X_PITCH, INPUT3_Y_PITCH, INPUT3_OFFSET },
            { INPUT4_SIZE_Y, INPUT4_SIZE_X, INPUT4_FEATURE_PITCH, INPUT4_X_PITCH, INPUT4_Y_PITCH, INPUT4_OFFSET }
        };

inline INPUT1_TYPE FUNC(accumulate)(INPUT1_TYPE acc, INPUT1_TYPE val) {
    return max(acc, val);
}

#define ACCUMULATOR_INIT_VAL INPUT1_VAL_MIN

KERNEL(pyramidROIAlign_gpu_ref)(
    const __global INPUT0_TYPE *boxes,
    const __global INPUT1_TYPE *P2,
    const __global INPUT2_TYPE *P3,
    const __global INPUT3_TYPE *P4,
    const __global INPUT4_TYPE *P5,
    __global OUTPUT_TYPE *output)
{
    const uint oyx = get_global_id(0);
    const uint ox = oyx % OUTPUT_SIZE_X;
    const uint oy = oyx / OUTPUT_SIZE_X;
    const uint of = get_global_id(1);
    const uint kerNum = (uint) get_global_id(2);

    INPUT0_TYPE hU = boxes[GET_DATA_INDEX(INPUT0, kerNum, 3, 0, 0)];
    INPUT0_TYPE hL = boxes[GET_DATA_INDEX(INPUT0, kerNum, 1, 0, 0)];
    INPUT0_TYPE h = hU - hL;
    INPUT0_TYPE wU = boxes[GET_DATA_INDEX(INPUT0, kerNum, 2, 0, 0)];
    INPUT0_TYPE wL = boxes[GET_DATA_INDEX(INPUT0, kerNum, 0, 0, 0)];
    INPUT0_TYPE w = wU - wL;

    // TODO This scale could be used when box coordinates are not normalized, but in pixel coordinates.
#ifdef PYRAMID_ROI_ALIGN_PIXEL_BOXES
    float image_area = IMAGE_SIZE_X * IMAGE_SIZE_Y;
    float scale = 1.f / sqrt(image_area);
#else
    float scale = 1.f;
#endif

    int roi_level = (int)round(PYRAMID_STARTING_LEVEL + log2(sqrt(h*w) * scale));
    // 0 <= roi_level < PYRAMID_LEVELS
    roi_level = min(PYRAMID_LEVELS - 1, max(0, roi_level));

    const __global INPUT1_TYPE* feature_map_ptrs[PYRAMID_LEVELS];

    feature_map_ptrs[0] = P2;
    feature_map_ptrs[1] = P3;
    feature_map_ptrs[2] = P4;
    feature_map_ptrs[3] = P5;

    const __global INPUT1_TYPE* feature_map_ptr = feature_map_ptrs[roi_level];

    const uint sampling_ratio_x = SAMPLING_RATIO_X != 0 ? SAMPLING_RATIO_X : (uint)ceil(1.f * w * IMAGE_SIZE_X / OUTPUT_SIZE_X);
    const uint sampling_ratio_y = SAMPLING_RATIO_Y != 0 ? SAMPLING_RATIO_Y : (uint)ceil(1.f * h * IMAGE_SIZE_Y / OUTPUT_SIZE_Y);

    //calculate cooficients for transformation
    INPUT0_TYPE y1 = hL * (parameters[roi_level].size_y - 1);
    INPUT0_TYPE x1 = wL * (parameters[roi_level].size_x - 1);
    INPUT0_TYPE y2 = hU * (parameters[roi_level].size_y - 1);
    INPUT0_TYPE x2 = wU * (parameters[roi_level].size_x - 1);
    INPUT0_TYPE deltaX = (x2 - x1) / (OUTPUT_SIZE_X);
    INPUT0_TYPE deltaY = (x2 - x1) / (OUTPUT_SIZE_Y);
    INPUT0_TYPE pool_deltaX = deltaX / sampling_ratio_x;
    INPUT0_TYPE pool_deltaY = deltaY / sampling_ratio_y;

    uint data_base_offset = parameters[roi_level].offset + parameters[roi_level].f_pitch * of;

    INPUT0_TYPE y_base = y1 + oy * deltaY + TO_INPUT0_TYPE(0.5f) * pool_deltaY;
    INPUT0_TYPE x_base = x1 + ox * deltaX + TO_INPUT0_TYPE(0.5f) * pool_deltaX;

    INPUT1_TYPE accumulator = ACCUMULATOR_INIT_VAL;

    //transformation
    for (int yi = 0; yi < sampling_ratio_y; ++yi) {
        INPUT0_TYPE y = y_base + yi * pool_deltaY;
        int y_low = (int)floor(y);
        int y_high = (int)ceil(y);

        y_low = clamp(y_low, 0, parameters[roi_level].size_y - 1);
        y_high = clamp(y_high, 0, parameters[roi_level].size_y - 1);

        if (y_low == y_high) {
            if (y_high + 1 <= parameters[roi_level].size_y)
                y_high += 1;
            else
                y_low -= 1;
        }

        INPUT0_TYPE y_high_coeff = y - y_low;
        INPUT0_TYPE y_low_coeff = y_high - y;

        for (int xi = 0; xi < sampling_ratio_x; ++xi) {
            INPUT0_TYPE x = x_base + xi * pool_deltaX;

            int x_left = (int)floor(x);
            int x_right = (int)ceil(x);

            x_left = clamp(x_left, 0, parameters[roi_level].size_x - 1);
            x_right = clamp(x_right, 0, parameters[roi_level].size_x - 1);

            if (x_left == x_right) {
                if (x_right + 1 <= parameters[roi_level].size_x)
                    x_right += 1;
                else
                    x_left -= 1;
            }

            INPUT0_TYPE x_right_coeff = x - x_left;
            INPUT0_TYPE x_left_coeff = x_right - x;

            uint low_left_idx = data_base_offset + parameters[roi_level].x_pitch * x_left + parameters[roi_level].y_pitch * y_low;
            uint high_left_idx = data_base_offset + parameters[roi_level].x_pitch * x_left + parameters[roi_level].y_pitch * y_high;
            uint low_right_idx = data_base_offset + parameters[roi_level].x_pitch * x_right + parameters[roi_level].y_pitch * y_low;
            uint high_right_idx = data_base_offset + parameters[roi_level].x_pitch * x_right + parameters[roi_level].y_pitch * y_high;

            INPUT1_TYPE low_left_val = feature_map_ptr[low_left_idx];
            INPUT1_TYPE high_left_val = feature_map_ptr[high_left_idx];
            INPUT1_TYPE low_right_val = feature_map_ptr[low_right_idx];
            INPUT1_TYPE high_right_val = feature_map_ptr[high_right_idx];

            INPUT1_TYPE left_val = y_low_coeff * low_left_val + y_high_coeff * high_left_val;
            INPUT1_TYPE right_val = y_low_coeff * low_right_val + y_high_coeff * high_right_val;

            INPUT1_TYPE interpolated_val = x_left_coeff * left_val + x_right_coeff * right_val;

            accumulator = FUNC_CALL(accumulate)(accumulator, interpolated_val);
        }
    }

    uint output_idx = GET_DATA_INDEX(OUTPUT, kerNum, of, oy, ox);
    output[output_idx] = TO_OUTPUT_TYPE(accumulator);
}
