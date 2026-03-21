// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(bevpool_v2_opt)(OPTIONAL_SHAPE_INFO_ARG
                       const __global INPUT0_TYPE* restrict cf,
                       const __global INPUT1_TYPE* restrict dw,
                       const __global INPUT2_TYPE* restrict idx,
                       const __global INPUT3_TYPE* restrict itv,
                       __global OUTPUT_TYPE* restrict output) {
    const uint block_idx = (uint)get_global_id(0);
    const uint interval_idx = (uint)get_global_id(1);
    const uint channel_base = block_idx * (uint)BLOCK_SIZE;

    if (channel_base >= OUTPUT_CHANNELS)
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

#if INPUT0_TYPE_SIZE == 2 && BLOCK_SIZE == 8
    float8 acc8 = (float8)(0.0f);
#elif INPUT0_TYPE_SIZE == 2 && BLOCK_SIZE == 4
    float4 acc4 = (float4)(0.0f);
#else
    float acc[BLOCK_SIZE] = {0.0f};
#endif

    for (int i = interval_start; i < interval_end; ++i) {
        const int dw_index_i = (int)idx[i];
        if (dw_index_i < 0 || (uint)dw_index_i >= INPUT1_LENGTH)
            continue;

        const uint dw_index = (uint)dw_index_i;
        const float dw_value = (float)dw[dw_index];
        const uint camera_idx = dw_index / depth_span;
        const uint feature_flat_idx = dw_index % feature_area;
        const uint cf_offset_base = (camera_idx * feature_area + feature_flat_idx) * INPUT_CHANNELS + channel_base;

#if INPUT0_TYPE_SIZE == 2 && BLOCK_SIZE == 8
        if (channel_base + 7 < OUTPUT_CHANNELS && cf_offset_base + 7 < INPUT0_LENGTH) {
            const __global half* cf_half = (const __global half*)cf;
            const half8 cf_vec = vload8(0, cf_half + cf_offset_base);
            acc8 = fma(convert_float8(cf_vec), (float8)(dw_value), acc8);
            continue;
        }
#elif INPUT0_TYPE_SIZE == 2 && BLOCK_SIZE == 4
        if (channel_base + 3 < OUTPUT_CHANNELS && cf_offset_base + 3 < INPUT0_LENGTH) {
            const __global half* cf_half = (const __global half*)cf;
            const half4 cf_vec = vload4(0, cf_half + cf_offset_base);
            acc4 = fma(convert_float4(cf_vec), (float4)(dw_value), acc4);
            continue;
        }
#endif

        for (uint lane = 0; lane < (uint)BLOCK_SIZE; ++lane) {
            const uint channel = channel_base + lane;
            if (channel >= OUTPUT_CHANNELS)
                break;

            const uint cf_offset = cf_offset_base + lane;
            if (cf_offset >= INPUT0_LENGTH)
                continue;

#if INPUT0_TYPE_SIZE == 2 && BLOCK_SIZE == 8
            acc8[lane] = fma((float)cf[cf_offset], dw_value, acc8[lane]);
#elif INPUT0_TYPE_SIZE == 2 && BLOCK_SIZE == 4
            acc4[lane] = fma((float)cf[cf_offset], dw_value, acc4[lane]);
#else
            acc[lane] = fma((float)cf[cf_offset], dw_value, acc[lane]);
#endif
        }
    }

    for (uint lane = 0; lane < (uint)BLOCK_SIZE; ++lane) {
        const uint channel = channel_base + lane;
        if (channel >= OUTPUT_CHANNELS)
            break;

        const int out_index_i = bev_linear + (int)(channel * FEATURE_WIDTH * FEATURE_HEIGHT);
        if (out_index_i < 0 || (uint)out_index_i >= OUTPUT_LENGTH)
            continue;

    #if INPUT0_TYPE_SIZE == 2 && BLOCK_SIZE == 8
        output[(uint)out_index_i] = (OUTPUT_TYPE)acc8[lane];
    #elif INPUT0_TYPE_SIZE == 2 && BLOCK_SIZE == 4
        output[(uint)out_index_i] = (OUTPUT_TYPE)acc4[lane];
    #else
        output[(uint)out_index_i] = (OUTPUT_TYPE)acc[lane];
    #endif
    }
}
