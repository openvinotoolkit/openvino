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

inline uint FUNC(get_input_index)(uint b, uint f, uint z, uint y, uint x)
{
#if INPUT0_DIMS < 5
    return INPUT0_GET_INDEX(b, f, y, x);
#elif INPUT0_DIMS == 5
    return INPUT0_GET_INDEX(b, f, z, y, x);
#else
#error [clDNN resample_ref.cl]: input format - not supported
#endif
}

inline uint FUNC(get_output_index)(uint b, uint f, uint z, uint y, uint x)
{
#if OUTPUT_DIMS < 5
    return OUTPUT_GET_INDEX(b, f, y, x);
#elif OUTPUT_DIMS == 5
    return OUTPUT_GET_INDEX(b, f, z, y, x);
#else
#error [clDNN resample_ref.cl]: output format - not supported
#endif
}


#define TRIANGLE_COEFF(x) (ACCUMULATOR_MAX_FUNC(ACCUMULATOR_VAL_ZERO, ACCUMULATOR_VAL_ONE - ACCUMULATOR_ABS_FUNC(x)))
#define unroll_for __attribute__((opencl_unroll_hint)) for

KERNEL (resample_gpu_ref)(__global INPUT0_TYPE* input,
                          __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                          , FUSED_OPS_DECLS
#endif
)
{
#if defined(SAMPLE_TYPE_NEAREST) && FEATURE_PACKED_MODE
    typedef MAKE_VECTOR_TYPE(INPUT0_TYPE, PACK_SIZE) in_pack_t;
    typedef MAKE_VECTOR_TYPE(OUTPUT_TYPE, PACK_SIZE) out_pack_t;

    const int ox = get_global_id(0);
#if OUTPUT_DIMS <= 4
    const int oy = get_global_id(1);
    const int oz = 0;
#else
    const int oy = (int)get_global_id(1) % OUTPUT_SIZE_Y;
    const int oz = (int)get_global_id(1) / OUTPUT_SIZE_Y;
#endif
    const int feature = ((int)get_global_id(2) * PACK_SIZE) % OUTPUT_FEATURE_NUM;
    const int batch = ((int)get_global_id(2) * PACK_SIZE) / OUTPUT_FEATURE_NUM;
    const int ix = floor(ox * X_RATIO);
    const int iy = floor(oy * Y_RATIO);
    const int iz = floor(oz * Z_RATIO);

    uint input_idx = FUNC_CALL(get_input_index)(batch, feature, iz, iy, ix);
    uint output_idx = FUNC_CALL(get_output_index)(batch, feature, oz, oy, ox);

    in_pack_t interp_val_pack = ((const __global in_pack_t*)(input + input_idx))[0];
    out_pack_t res;
    unroll_for (uint pi = 0; pi < PACK_SIZE; ++pi) {
        INPUT0_TYPE interp_val = interp_val_pack[pi];
    #if HAS_FUSED_OPS
        #define OF_ID (feature + pi)
        FUSED_OPS;
        res[pi] = FUSED_OPS_RESULT;
    #else
        res[pi] = ACTIVATION(interp_val, ACTIVATION_PARAMS);
    #endif
    }
    ((__global out_pack_t*)(output + output_idx))[0] = res;

#elif defined(SAMPLE_TYPE_NEAREST)
    const int ox = get_global_id(0);
#if OUTPUT_DIMS <= 4
    const int oy = get_global_id(1);
    const int oz = 0;
#else
    const int oy = (int)get_global_id(1) % OUTPUT_SIZE_Y;
    const int oz = (int)get_global_id(1) / OUTPUT_SIZE_Y;
#endif
    const int feature = (int)get_global_id(2) % OUTPUT_FEATURE_NUM;
    const int batch = (int)get_global_id(2) / OUTPUT_FEATURE_NUM;
    const int ix = floor(ox * X_RATIO);
    const int iy = floor(oy * Y_RATIO);
    const int iz = floor(oz * Z_RATIO);

    INPUT0_TYPE interp_val = input[FUNC_CALL(get_input_index)(batch, feature, iz, iy, ix)];
#if HAS_FUSED_OPS
    #define OF_ID (feature)
    FUSED_OPS;
    OUTPUT_TYPE res = FUSED_OPS_RESULT;
#else
    OUTPUT_TYPE res = ACTIVATION(interp_val, ACTIVATION_PARAMS);
#endif
    output[FUNC_CALL(get_output_index)(batch, feature, oz, oy, ox)] = res;

#elif defined(SAMPLE_TYPE_INTERP)
    const int ox = get_global_id(0);
    const int oy = get_global_id(1);
    const int feature = 0;
    const int batch = get_global_id(2);
    const float ix = X_RATIO * ox;
    const float iy = Y_RATIO * oy;

#ifdef LEFTOVERS
    if (ox >= OUTPUT_SIZE_X)
        return;
#endif

    const int top_y_index    = (int)(floor(iy));
    const int bottom_y_index = min((int)ceil(iy), INPUT0_SIZE_Y - 1);
    const int left_x_index   = (int)(floor(ix));
    const int right_x_index  = min((int)ceil(ix), INPUT0_SIZE_X - 1);

    const ACCUMULATOR_TYPE dx = TO_ACCUMULATOR_TYPE(ix - left_x_index);
    const ACCUMULATOR_TYPE dy = TO_ACCUMULATOR_TYPE(iy - top_y_index);

    unroll_for(int in_f = 0; in_f < OUTPUT_FEATURE_NUM; in_f++) {
        INPUT0_TYPE top_left = input[INPUT0_GET_INDEX(batch, in_f, top_y_index, left_x_index)];
        INPUT0_TYPE top_right = input[INPUT0_GET_INDEX(batch, in_f, top_y_index, right_x_index)];
        INPUT0_TYPE bottom_left = input[INPUT0_GET_INDEX(batch, in_f, bottom_y_index, left_x_index)];
        INPUT0_TYPE bottom_right = input[INPUT0_GET_INDEX(batch, in_f, bottom_y_index, right_x_index)];

        ACCUMULATOR_TYPE top = TO_ACCUMULATOR_TYPE(top_left) + (TO_ACCUMULATOR_TYPE(top_right) - TO_ACCUMULATOR_TYPE(top_left)) * dx;
        ACCUMULATOR_TYPE bottom = TO_ACCUMULATOR_TYPE(bottom_left) + (TO_ACCUMULATOR_TYPE(bottom_right) - TO_ACCUMULATOR_TYPE(bottom_left)) * dx;

        ACCUMULATOR_TYPE interp_val = top + (bottom - top) * dy;

#if HAS_FUSED_OPS
        #define OF_ID (in_f)
        FUSED_OPS;
        OUTPUT_TYPE res = FUSED_OPS_RESULT;
#else
        OUTPUT_TYPE res = TO_OUTPUT_TYPE(ACTIVATION(interp_val, ACTIVATION_PARAMS));
#endif
        output[OUTPUT_GET_INDEX(batch, in_f, oy, ox)] = res;
    }
#elif defined(SAMPLE_TYPE_CAFFE_INTERP)
    const int ox = (int)get_global_id(0) % OUTPUT_SIZE_X;
    const int oy = (int)get_global_id(0) / OUTPUT_SIZE_X;
    const int feature_block_nun = get_global_id(1);
    const int feature = feature_block_nun * FEATURE_BLOCK_SIZE;
#if OUTPUT_DIMS <= 4
    const int batch = get_global_id(2);
    const int oz = 0;
#else
    const int batch = (int)get_global_id(2) % OUTPUT_BATCH_NUM;
    const int oz    = (int)get_global_id(2) / OUTPUT_BATCH_NUM;
#endif

    const ACCUMULATOR_TYPE ix = ox * X_RATIO + X_RATIO_HALF - 0.5f;
    const ACCUMULATOR_TYPE iy = oy * Y_RATIO + Y_RATIO_HALF - 0.5f;
    const ACCUMULATOR_TYPE iz = oz * Z_RATIO + Z_RATIO_HALF - 0.5f;

    const int ix_r = (int)ix;
    const int iy_r = (int)iy;
    const int iz_r = (int)iz;

#if ANTIALIAS == 1
    const ACCUMULATOR_TYPE ax = 1.0f / X_RATIO;
    const ACCUMULATOR_TYPE ay = 1.0f / Y_RATIO;
    const ACCUMULATOR_TYPE az = 1.0f / Z_RATIO;
#else
    const ACCUMULATOR_TYPE ax = 1.0f;
    const ACCUMULATOR_TYPE ay = 1.0f;
    const ACCUMULATOR_TYPE az = 1.0f;
#endif
    const int rx = (X_RATIO < 1.0f) ? 2 : (int)ceil(TO_ACCUMULATOR_TYPE(KERNEL_W) / ax);
    const int ry = (Y_RATIO < 1.0f) ? 2 : (int)ceil(TO_ACCUMULATOR_TYPE(KERNEL_W) / ay);
    const int rz = (Z_RATIO < 1.0f) ? 2 : (int)ceil(TO_ACCUMULATOR_TYPE(KERNEL_W) / az);

    ACCUMULATOR_TYPE sum[FEATURE_BLOCK_SIZE];
    for (int i = 0; i < FEATURE_BLOCK_SIZE; i++)
        sum[i] = 0;

    ACCUMULATOR_TYPE wsum = 0;

    int const y_init = max(0, iy_r - ry);
    int const x_init = max(0, ix_r - rx);
    int const z_init = max(0, iz_r - rz);
    int const y_max = min(INPUT0_SIZE_Y, iy_r + ry + 1);
    int const x_max = min(INPUT0_SIZE_X, ix_r + rx + 1);
    int const z_max = min(INPUT0_SIZE_Z, iz_r + rz + 1);

    unroll_for(int z = z_init; z < z_max; z++) {
        unroll_for(int y = y_init; y < y_max; y++) {
            unroll_for(int x = x_init; x < x_max; x++) {
                ACCUMULATOR_TYPE dx = ix - x;
                ACCUMULATOR_TYPE dy = iy - y;
                ACCUMULATOR_TYPE dz = iz - z;
#if ANTIALIAS == 1
                ACCUMULATOR_TYPE w = ax * TRIANGLE_COEFF(ax * dx) * ay * TRIANGLE_COEFF(ay * dy) * az * triangleCoeff(az * dz);
#else
                ACCUMULATOR_TYPE w = TRIANGLE_COEFF(dx) * TRIANGLE_COEFF(dy) * TRIANGLE_COEFF(dz);
#endif

#ifndef LEFTOVERS
                unroll_for(int f = 0; f < FEATURE_BLOCK_SIZE; f++) {
#else
                const int f_max = min(FEATURE_BLOCK_SIZE, FEATURE_LEFTOVER);
                unroll_for(int f = 0; f < f_max; f++) {
#endif
                if (w != 0)
                    sum[f] += w * TO_ACCUMULATOR_TYPE(input[FUNC_CALL(get_input_index)(batch, feature + f, z, y, x)]);
                }
                wsum += w;
            }
        }
    }
#ifndef LEFTOVERS
    unroll_for (int f = 0; f < FEATURE_BLOCK_SIZE; f++) {
#else
    const int f_max = min(FEATURE_BLOCK_SIZE, FEATURE_LEFTOVER);
    unroll_for (int f = 0; f < f_max; f++) {
#endif

        ACCUMULATOR_TYPE interp_val = (wsum == 0) ? 0 : (sum[f] / wsum);
#if HAS_FUSED_OPS
        #define OF_ID (feature + f)
        FUSED_OPS;
        OUTPUT_TYPE res = FUSED_OPS_RESULT;
#else
        OUTPUT_TYPE res = TO_OUTPUT_TYPE(ACTIVATION(interp_val, ACTIVATION_PARAMS));
#endif
        output[FUNC_CALL(get_output_index)(batch, feature + f, oz, oy, ox)] = res;
    }
#endif
}

#undef unroll_for
#undef TRIANGLE_COEFF
