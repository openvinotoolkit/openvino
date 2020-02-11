// Copyright (C) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/common.cl"
#include "include/data_types.cl"
#include "include/include_all.cl"

#define TRIANGLE_COEFF(x) (INPUT0_MAX_FUNC(INPUT0_VAL_ZERO, INPUT0_VAL_ONE - INPUT0_ABS_FUNC(x)))
#define unroll_for __attribute__((opencl_unroll_hint)) for

KERNEL (resample_gpu_ref)(__global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
#if defined(SAMPLE_TYPE_NEAREST)
    const int ox = get_global_id(0);
    const int oy = get_global_id(1);
    const int feature = get_global_id(2) % OUTPUT_FEATURE_NUM;
    const int batch = get_global_id(2) / OUTPUT_FEATURE_NUM;
    const int ix = floor(ox * X_RATIO);
    const int iy = floor(oy * Y_RATIO);
    output[OUTPUT_GET_INDEX(batch, feature, oy, ox)] = ACTIVATION(input[INPUT0_GET_INDEX(batch, feature, iy, ix)], ACTIVATION_PARAMS);
#elif defined(SAMPLE_TYPE_INTERP)
    const int ox = get_global_id(0);
    const int oy = get_global_id(1);
    const int feature = 0;
    const int batch = get_global_id(2);
    const INPUT0_TYPE ix = TO_INPUT0_TYPE(X_RATIO) * ox;
    const INPUT0_TYPE iy = TO_INPUT0_TYPE(Y_RATIO) * oy;

#ifdef LEFTOVERS
    if (ox >= OUTPUT_SIZE_X)
        return;
#endif

    const int top_y_index    = (int)(floor(iy));
    const int bottom_y_index = (int)(min(ceil(iy), TO_INPUT0_TYPE(INPUT0_SIZE_Y) - 1));
    const int left_x_index   = (int)(floor(ix));
    const int right_x_index  = (int)(min(ceil(ix), TO_INPUT0_TYPE(INPUT0_SIZE_X) - 1));

    const INPUT0_TYPE dx = ix - left_x_index;
    const INPUT0_TYPE dy = iy - top_y_index;

    unroll_for (int in_f = 0; in_f < OUTPUT_FEATURE_NUM; in_f++) {
        INPUT0_TYPE top_left     = input[INPUT0_GET_INDEX(batch, in_f, top_y_index, left_x_index)];
        INPUT0_TYPE top_right    = input[INPUT0_GET_INDEX(batch, in_f, top_y_index, right_x_index)];
        INPUT0_TYPE bottom_left  = input[INPUT0_GET_INDEX(batch, in_f, bottom_y_index, left_x_index)];
        INPUT0_TYPE bottom_right = input[INPUT0_GET_INDEX(batch, in_f, bottom_y_index, right_x_index)];

        INPUT0_TYPE top    = top_left + (top_right - top_left) * dx;
        INPUT0_TYPE bottom = bottom_left + (bottom_right - bottom_left) * dx;

        output[OUTPUT_GET_INDEX(batch, in_f, oy, ox)] = ACTIVATION(top + (bottom - top) * dy, ACTIVATION_PARAMS);
    }
#elif defined(SAMPLE_TYPE_CAFFE_INTERP)
    const int ox = get_global_id(0) % OUTPUT_SIZE_X;
    const int oy = get_global_id(0) / OUTPUT_SIZE_X;
    const int feature_block_nun = get_global_id(1);
    const int feature = feature_block_nun * FEATURE_BLOCK_SIZE;
    const int batch = get_global_id(2);

    __global INPUT0_TYPE* input_ptr = input + batch * INPUT0_BATCH_PITCH + feature * INPUT0_FEATURE_PITCH;
    __global OUTPUT_TYPE* output_ptr = output + batch * OUTPUT_BATCH_PITCH + feature * OUTPUT_FEATURE_PITCH;

    const INPUT0_TYPE ix = ox * X_RATIO + Y_RATIO_HALF - 0.5f;
    const INPUT0_TYPE iy = oy * Y_RATIO + X_RATIO_HALF - 0.5f;

    const int ix_r = (int)ix;
    const int iy_r = (int)iy;

#if ANTIALIAS == 1
    const INPUT0_TYPE ax = 1.0f / X_RATIO;
    const INPUT0_TYPE ay = 1.0f / Y_RATIO;
#else
    const INPUT0_TYPE ax = 1.0f;
    const INPUT0_TYPE ay = 1.0f;
#endif
    const int rx = (X_RATIO < 1.0f) ? 2 : (int)ceil(TO_INPUT0_TYPE(KERNEL_W) / ax);
    const int ry = (Y_RATIO < 1.0f) ? 2 : (int)ceil(TO_INPUT0_TYPE(KERNEL_W) / ay);

    INPUT0_TYPE sum[FEATURE_BLOCK_SIZE];
    for (int i = 0; i < FEATURE_BLOCK_SIZE; i++)
        sum[i] = 0;

    INPUT0_TYPE wsum = 0;

    int const y_init = max(0, iy_r - ry);
    int const x_init = max(0, ix_r - rx);
    int const y_max = min(INPUT0_SIZE_Y, iy_r + ry + 1);
    int const x_max = min(INPUT0_SIZE_X, ix_r + rx + 1);

    unroll_for (int y = y_init; y < y_max; y++) {
        unroll_for (int x = x_init; x < x_max; x++) {
            INPUT0_TYPE dx = ix - x;
            INPUT0_TYPE dy = iy - y;
#if ANTIALIAS == 1
            INPUT0_TYPE w = ax * TRIANGLE_COEFF(ax * dx) * ay * TRIANGLE_COEFF(ay * dy);
#else
            INPUT0_TYPE w = TRIANGLE_COEFF(dx) * TRIANGLE_COEFF(dy);
#endif

#ifndef LEFTOVERS
            unroll_for (int f = 0; f < FEATURE_BLOCK_SIZE; f++) {
#else
            const int f_max = min(FEATURE_BLOCK_SIZE, FEATURE_LEFTOVER);
            unroll_for (int f = 0; f < f_max; f++) {
#endif
                if (w != 0)
                    sum[f] += w * input_ptr[f * INPUT0_FEATURE_PITCH + y * INPUT0_Y_PITCH + x];
            }
            wsum += w;
        }
    }
#ifndef LEFTOVERS
    unroll_for (int f = 0; f < FEATURE_BLOCK_SIZE; f++) {
#else
    const int f_max = min(FEATURE_BLOCK_SIZE, FEATURE_LEFTOVER);
    unroll_for (int f = 0; f < f_max; f++) {
#endif
        output_ptr[f * OUTPUT_FEATURE_PITCH + oy * OUTPUT_Y_PITCH + ox] = ACTIVATION((wsum == 0) ? 0 : (sum[f] / wsum), ACTIVATION_PARAMS);
    }
#endif
}

#undef unroll_for
#undef TRIANGLE_COEFF
