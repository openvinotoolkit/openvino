// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

inline OUTPUT_TYPE FUNC(clip_great)(OUTPUT_TYPE x, OUTPUT_TYPE threshold) {
    return x < threshold ? x : threshold;
}

inline OUTPUT_TYPE FUNC(clip_less)(OUTPUT_TYPE x, OUTPUT_TYPE threshold) {
    return x > threshold ? x : threshold;
}

inline uint FUNC(get_index)(INPUT0_TYPE w, INPUT0_TYPE h) {
    return (w + h * WIDTH) * NUM_PRIORS_4;
}

inline void FUNC(calculate_data)(OUTPUT_TYPE center_x,
                                 OUTPUT_TYPE center_y,
                                 OUTPUT_TYPE box_width,
                                 OUTPUT_TYPE box_height,
                                 bool clip,
                                 uint* output_index,
                                 __global OUTPUT_TYPE* dst_data) {
    uint idx = *output_index;
    if (clip) {
        // order: xmin, ymin, xmax, ymax
        dst_data[OUTPUT_GET_INDEX(0, idx++, 0, 0)] = FUNC_CALL(clip_less)((center_x - box_width) * IWI, 0);
        dst_data[OUTPUT_GET_INDEX(0, idx++, 0, 0)] = FUNC_CALL(clip_less)((center_y - box_height) * IHI, 0);
        dst_data[OUTPUT_GET_INDEX(0, idx++, 0, 0)] = FUNC_CALL(clip_great)((center_x + box_width) * IWI, 1);
        dst_data[OUTPUT_GET_INDEX(0, idx++, 0, 0)] = FUNC_CALL(clip_great)((center_y + box_height) * IHI, 1);
    } else {
        dst_data[OUTPUT_GET_INDEX(0, idx++, 0, 0)] = (center_x - box_width) * IWI;
        dst_data[OUTPUT_GET_INDEX(0, idx++, 0, 0)] = (center_y - box_height) * IHI;
        dst_data[OUTPUT_GET_INDEX(0, idx++, 0, 0)] = (center_x + box_width) * IWI;
        dst_data[OUTPUT_GET_INDEX(0, idx++, 0, 0)] = (center_y + box_height) * IHI;

        *output_index = idx;
    }
};

KERNEL(ref)
(const __global INPUT0_TYPE* output_size, const __global INPUT1_TYPE* image_size, __global OUTPUT_TYPE* output) {
    const uint w = get_global_id(0);
    const uint h = get_global_id(1);
    uint out_index = FUNC_CALL(get_index)(w, h);
    const uint start_out_index = out_index;

    OUTPUT_TYPE center_x, center_y;
    #ifdef STEP
        center_x = (OFFSET + w) * STEP;
        center_y = (OFFSET + h) * STEP;
    #else
        center_x = (w + 0.5f) * STEP_X;
        center_y = (h + 0.5f) * STEP_Y;
    #endif
    OUTPUT_TYPE box_width, box_height;

    for (uint s = 0; s < FIXED_SIZE_SIZE; ++s) {
        #if FIXED_SIZE_SIZE > 0
            OUTPUT_TYPE fixed_size_ = FIXED_SIZE[s];
        #else
            OUTPUT_TYPE fixed_size_ = 0;
        #endif

        box_height = box_width = fixed_size_ * 0.5f;

        #if FIXED_RATIO_SIZE > 0
            for (uint k = 0; k < FIXED_RATIO_SIZE; ++k) {
                OUTPUT_TYPE ar = FIXED_RATIO[k];
                uint density_ = DENSITY[s];
                uint shift = FIXED_SIZE[s] / density_;
                ar = sqrt(ar);
                OUTPUT_TYPE box_width_ratio = FIXED_SIZE[s] * 0.5f * ar;
                OUTPUT_TYPE box_height_ratio = FIXED_SIZE[s] * 0.5f / ar;
                for (uint r = 0; r < density_; ++r) {
                    for (uint c = 0; c < density_; ++c) {
                        OUTPUT_TYPE center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                        OUTPUT_TYPE center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;
                        FUNC_CALL(calculate_data)
                        (center_x_temp, center_y_temp, box_width_ratio, box_height_ratio, true, &out_index, output);
                    }
                }
            }
        #else
            #if DENSITY_SIZE > 0 && FIXED_SIZE_SIZE > 0
                uint density_ = DENSITY[s];
                uint shift = FIXED_SIZE[s] / density_;
                for (uint r = 0; r < density_; ++r) {
                    for (uint c = 0; c < density_; ++c) {
                        OUTPUT_TYPE center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                        OUTPUT_TYPE center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;
                        FUNC_CALL(calculate_data)
                        (center_x_temp, center_y_temp, box_width, box_height, true, &out_index, output);
                    }
                }
            #endif
            //  Rest of priors
            for (uint k = 0; k < ASPECT_RATIO_SIZE; ++k) {
                OUTPUT_TYPE ar = ASPECT_RATIO[k];
                if (fabs(ar - 1.) < 1e-6) {
                    continue;
                }
                #if DENSITY_SIZE > 0 && FIXED_SIZE_SIZE > 0
                    uint density_ = DENSITY[s];
                    uint shift = FIXED_SIZE[s] / density_;
                    ar = sqrt(ar);
                    OUTPUT_TYPE box_width_ratio = FIXED_SIZE[s] * 0.5f * ar;
                    OUTPUT_TYPE box_height_ratio = FIXED_SIZE[s] * 0.5f / ar;
                    for (uint r = 0; r < density_; ++r) {
                        for (uint c = 0; c < density_; ++c) {
                            OUTPUT_TYPE center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                            OUTPUT_TYPE center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;
                            FUNC_CALL(calculate_data)
                            (center_x_temp, center_y_temp, box_width_ratio, box_height_ratio, true, &out_index, output);
                        }
                    }
                #endif
            }
        #endif
    }

    for (uint ms_idx = 0; ms_idx < MIN_SIZE_SIZE; ++ms_idx) {
        box_width = MIN_SIZE[ms_idx] * 0.5f;
        box_height = MIN_SIZE[ms_idx] * 0.5f;
        FUNC_CALL(calculate_data)(center_x, center_y, box_width, box_height, false, &out_index, output);
        #ifdef MIN_MAX_ASPECT_RATIO_ORDER
            if (MAX_SIZE_SIZE > ms_idx) {
                box_width = box_height = sqrt(MIN_SIZE[ms_idx] * MAX_SIZE[ms_idx]) * 0.5f;
                FUNC_CALL(calculate_data)(center_x, center_y, box_width, box_height, false, &out_index, output);

            }

            if (SCALE_ALL_SIZES || (!SCALE_ALL_SIZES && (ms_idx == MIN_SIZE_SIZE - 1))) {
                uint s_idx = SCALE_ALL_SIZES ? ms_idx : 0;
                for (uint k = 0; k < ASPECT_RATIO_SIZE; ++k) {
                    OUTPUT_TYPE ar = ASPECT_RATIO[k];
                    if (fabs(ar - 1.0f) < 1e-6) {
                        continue;
                    }

                    ar = sqrt(ar);
                    box_width = MIN_SIZE[s_idx] * 0.5f * ar;
                    box_height = MIN_SIZE[s_idx] * 0.5f / ar;
                    FUNC_CALL(calculate_data)(center_x, center_y, box_width, box_height, false, &out_index, output);
                }
            }
        #else
            if (SCALE_ALL_SIZES || (!SCALE_ALL_SIZES && (ms_idx == MIN_SIZE_SIZE - 1))) {
                uint s_idx = SCALE_ALL_SIZES ? ms_idx : 0;
                for (uint k = 0; k < ASPECT_RATIO_SIZE; ++k) {
                    OUTPUT_TYPE ar = ASPECT_RATIO[k];
                    if (fabs(ar - 1.0f) < 1e-6) {
                        continue;
                    };

                    ar = sqrt(ar);
                    box_width = MIN_SIZE[s_idx] * 0.5f * ar;
                    box_height = MIN_SIZE[s_idx] * 0.5f / ar;
                    FUNC_CALL(calculate_data)(center_x, center_y, box_width, box_height, false, &out_index, output);
                }
            }

            if (MAX_SIZE_SIZE > ms_idx) {
                box_width = box_height = sqrt(MIN_SIZE[ms_idx] * MAX_SIZE[ms_idx]) * 0.5f;
                FUNC_CALL(calculate_data)(center_x, center_y, box_width, box_height, false, &out_index, output);
            }
        #endif
    }

    #ifdef CLIP
        for (uint i = start_out_index; i < out_index; ++i) {
            const uint out_idx = OUTPUT_GET_INDEX(0, 0, 0, i);
            output[out_idx] = (min)((max)(output[out_idx], 0.0f), 1.0f);
        }
    #endif

    const uint channel_size = OUTPUT_LENGTH / 2;
    #if VARIANCE_SIZE == 1
        for (uint i = start_out_index; i < out_index; ++i) {
            output[OUTPUT_GET_INDEX(1, i, 0, 0)] = VARIANCE[0];
        }
    #elif VARIANCE_SIZE == 4
        for (uint i = start_out_index; i < out_index; ++i) {
            for (uint j = 0; j < 4; ++j) {
                output[OUTPUT_GET_INDEX(1, i * 4 + j, 0, 0)] = VARIANCE[j];
            }
        }
    #else
        #error Invalid Variances size
    #endif
}
