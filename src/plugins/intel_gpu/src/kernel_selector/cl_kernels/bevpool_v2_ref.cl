// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(bevpool_v2_ref)(OPTIONAL_SHAPE_INFO_ARG
                       const __global INPUT0_TYPE* restrict cf,
#if INPUTS_COUNT > 1
                       const __global INPUT1_TYPE* restrict dw,
#endif
#if INPUTS_COUNT > 2
                       const __global INPUT2_TYPE* restrict idx,
#endif
#if INPUTS_COUNT > 3
                       const __global INPUT3_TYPE* restrict itv,
#endif
                       __global OUTPUT_TYPE* restrict output) {
        const uint channel_idx = (uint)get_global_id(0);
        const uint interval_idx = (uint)get_global_id(1);

        if (channel_idx >= OUTPUT_CHANNELS)
        return;

        const uint interval_count = (uint)(INPUT3_LENGTH / 3);
        if (interval_idx >= interval_count)
                return;

        const uint interval_base = interval_idx * 3;
        const int interval_start = (int)itv[interval_base + 0];
        const int interval_end = (int)itv[interval_base + 1];
        const int bev_linear = (int)itv[interval_base + 2];

        const uint depth_bins = (uint)((D_BOUND_MAX - D_BOUND_MIN) / D_BOUND_STEP);
        const uint feature_area = IMAGE_WIDTH * IMAGE_HEIGHT;
        const uint depth_span = depth_bins * feature_area;

        float acc = 0.0f;
        for (int i = interval_start; i < interval_end; ++i) {
                const int dw_index_i = (int)idx[i];
                if (dw_index_i < 0 || (uint)dw_index_i >= INPUT1_LENGTH)
                        continue;

                const uint dw_index = (uint)dw_index_i;
                const uint camera_idx = dw_index / depth_span;
                const uint feature_flat_idx = dw_index % feature_area;

                const uint cf_offset = (camera_idx * feature_area + feature_flat_idx) * INPUT_CHANNELS + channel_idx;
                if (cf_offset >= INPUT0_LENGTH)
                        continue;

                acc = fma((float)cf[cf_offset], (float)dw[dw_index], acc);
    }

        const int out_index_i = bev_linear + (int)(channel_idx * FEATURE_WIDTH * FEATURE_HEIGHT);
        if (out_index_i < 0 || (uint)out_index_i >= OUTPUT_LENGTH)
                return;

        output[(uint)out_index_i] = (OUTPUT_TYPE)acc;
}
