// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

KERNEL(histc_ref)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output) {
    if (get_global_id(0) != 0)
        return;

    for (uint i = 0; i < BINS; ++i) {
        output[i] = (OUTPUT_TYPE)0;
    }

    if (TOTAL_ELEMENTS == 0 || BINS == 0)
        return;

    float range_min = MIN_VAL;
    float range_max = MAX_VAL;
    if (range_min == 0.0f && range_max == 0.0f) {
        range_min = (float)input[0];
        range_max = (float)input[0];
        for (uint i = 1; i < TOTAL_ELEMENTS; ++i) {
            const float v = (float)input[i];
            if (v < range_min)
                range_min = v;
            if (v > range_max)
                range_max = v;
        }
    }

    if (range_min == range_max) {
        for (uint i = 0; i < TOTAL_ELEMENTS; ++i) {
            const float v = (float)input[i];
            if (v >= range_min && v <= range_max) {
                output[0] += (OUTPUT_TYPE)1;
            }
        }
        return;
    }

    const float step = (range_max - range_min) / (float)BINS;
    for (uint i = 0; i < TOTAL_ELEMENTS; ++i) {
        const float v = (float)input[i];
        if (v >= range_min && v <= range_max) {
            int bin = (int)((v - range_min) / step);
            if (bin >= BINS)
                bin = BINS - 1;
            output[bin] += (OUTPUT_TYPE)1;
        }
    }
}
