/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "gpu/ocl/ocl_types.h"

#undef SRC_OFF
#undef DST_OFF

#define SRC_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC, x0, x1, x2, x3, x4, x5)
#define DST_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(DST, x0, x1, x2, x3, x4, x5)
#define STAT_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(STAT, x0, x1, x2, x3, x4, x5)

#if IS_FWD
#if VECTORIZE_CALC_STATS == 1
KERNEL_ATTR
__kernel void ref_lnorm_fwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *dst, __global float *scale,
        __global float *shift, float eps) {

    int x[6] = {0};
    x[0] = GWS_GET_X0();
    x[1] = GWS_GET_X1();
    x[2] = GWS_GET_X2();
    x[3] = GWS_GET_X3();

    if (x[0] >= DST_D0 || x[1] >= DST_D1 || x[2] >= DST_D2 || x[3] >= DST_D3) {
        int local_id = get_sub_group_local_id();
        for (int c = 0; c < C; c += SUB_GROUP_SIZE) {
            x[NDIMS - 1] = c + local_id;
            int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
            dst[dst_off] = CONVERT_DATA_T(0.f);
        }
        return;
    }

    int s_off = STAT_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

    float v_mean = CALCULATE_STATS ? 0 : mean[s_off];
    float v_variance = CALCULATE_STATS ? 0 : variance[s_off];

    if (CALCULATE_STATS) {
        VECT_FLOAT_T v_acc = 0;
        for (int c = 0; c < C; c += SUB_GROUP_SIZE * VECT_DT_N) {
            x[NDIMS - 1] = c;
            int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
            v_acc += CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                    (const __global BLOCK_DATA_T *)&src[src_off])));
        }
#if VECT_DT_N == 1
        v_mean = v_acc;
#else // VECT_DT_N == 1
        v_mean = 0;
        for (int i = 0; i < VECT_DT_N; ++i) {
            v_mean += v_acc[i];
        }
#endif // VECT_DT_N == 1

        float total_sum = sub_group_reduce_add(v_mean);
        v_mean = total_sum / C;

        v_acc = 0;
        VECT_FLOAT_T m = 0;

        for (int c = 0; c < C; c += SUB_GROUP_SIZE * VECT_DT_N) {
            x[NDIMS - 1] = c;
            int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

            m = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                    (const __global BLOCK_DATA_T *)&src[src_off])));
            m -= v_mean;
            v_acc += m * m;
        }
#if VECT_DT_N == 1
        v_variance = v_acc;
#else // VECT_DT_N == 1
        v_variance = 0;
        for (int i = 0; i < VECT_DT_N; ++i) {
            v_variance += v_acc[i];
        }
#endif // VECT_DT_N == 1

        total_sum = sub_group_reduce_add(v_variance);
        v_variance = total_sum / C;
    }
    float sqrt_variance = sqrt(v_variance + eps);
    int local_id = get_sub_group_local_id();
    for (int c = 0; c < C; c += SUB_GROUP_SIZE) {
        float sm = (scale ? scale[c + local_id] : 1.0f) / sqrt_variance;
        float sv = shift ? shift[SHIFT_OFF + c + local_id] : 0.0f;

        x[NDIMS - 1] = c + local_id;
        int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
        int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

        dst[dst_off] = TO_DST(sm * (SRC_TO_REF(src[src_off]) - v_mean) + sv);
    }

    if (CALCULATE_STATS) {
        if (SAVE_STATS) {
            mean[s_off] = v_mean;
            variance[s_off] = v_variance;
        }
    }
}
#else //VECTORIZE_CALC_STATS == 1

KERNEL_ATTR
__kernel void ref_lnorm_fwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *dst, __global float *scale,
        __global float *shift, float eps) {

    int x[6] = {0};
    x[0] = GWS_GET_X0();
    x[1] = GWS_GET_X1();
    x[2] = GWS_GET_X2();
    x[3] = GWS_GET_X3();

    if (x[0] >= DST_D0 || x[1] >= DST_D1 || x[2] >= DST_D2 || x[3] >= DST_D3) {
        for (int c = 0; c < C; ++c) {
            x[NDIMS - 1] = c;
            int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
            dst[dst_off] = CONVERT_DATA_T(0.f);
        }
        return;
    }

    int s_off = STAT_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

    float v_mean = CALCULATE_STATS ? 0 : mean[s_off];
    float v_variance = CALCULATE_STATS ? 0 : variance[s_off];

    if (CALCULATE_STATS) {
        for (int c = 0; c < C; ++c) {
            x[NDIMS - 1] = c;
            int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

            v_mean += SRC_TO_REF(src[src_off]);
        }
        v_mean /= C;

        for (int c = 0; c < C; ++c) {
            x[NDIMS - 1] = c;
            int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

            float m = SRC_TO_REF(src[src_off]) - v_mean;
            v_variance += m * m;
        }
        v_variance /= C;
    }

    float sqrt_variance = sqrt(v_variance + eps);
    for (int c = 0; c < C; ++c) {
        float sm = (scale ? scale[c] : 1.0f) / sqrt_variance;
        float sv = shift ? shift[SHIFT_OFF + c] : 0.0f;

        x[NDIMS - 1] = c;
        int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
        int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

        dst[dst_off] = TO_DST(sm * (SRC_TO_REF(src[src_off]) - v_mean) + sv);
    }

    if (CALCULATE_STATS) {
        if (SAVE_STATS) {
            mean[s_off] = v_mean;
            variance[s_off] = v_variance;
        }
    }
}
#endif
#endif

#if IS_BWD
#if USE_SCALESHIFT || USE_SCALE || USE_SHIFT
#if VECTORIZE_BWD_SCALESHIFT

#if VECTOR_SIZE_SCALESHIFT == 1
#define VECTORIZED_VERSION(x) x
#define vector_load(x) (x);
#else
#define VECTORIZED_VERSION(x) CONCAT2(x, VECTOR_SIZE_SCALESHIFT)
#define vector_load(x) CONCAT2(vload, VECTOR_SIZE_SCALESHIFT)(0, &x);
#endif

#if DT_BF16 == 1
#define convert_vector_to_float cvt_bf16_to_f32
#else
#define convert_vector_to_float VECTORIZED_VERSION(convert_float)
#endif

#define as_vector_data_t VECTORIZED_VERSION(AS_DATA_T)
#define sub_group_read VECTORIZED_VERSION(BLOCK_READ)
#define vector_float VECTORIZED_VERSION(float)

NAMED_KERNEL_ATTR(SCALESHIFT)
__kernel void ref_lnorm_bwd_scaleshift(__global DATA_T *src,
        __global float *mean, __global float *variance,
        __global DATA_T *diff_dst, __global float *diff_scale,
        __global float *diff_shift, float eps) {

    const int c = GWS_GET_C();
    const int n_chunk_idx = GWS_GET_N();
    const int n_start = n_chunk_idx * N_CHUNK_SIZE;
    const int n_end = n_start + N_CHUNK_SIZE;

    // diff_scale and diff_shift use the same tensor in scratchpad
    const int shift_off = N_CHUNKS * C;
    diff_shift += shift_off;

    vector_float diff_gamma_vect = 0;
    vector_float diff_beta_vect = 0;

    for (int n_off = n_start; n_off < n_end; n_off += VECTOR_SIZE_SCALESHIFT) {
        const vector_float mean_vect = vector_load(mean[n_off]);
        const vector_float variance_vect = vector_load(variance[n_off]);
        const vector_float inv_sqrt_variance = 1.0f / sqrt(variance_vect + eps);
#if NDIMS == 2
        const int src_off = SRC_OFF(n_off, c, 0, 0, 0, 0);
        const int dst_off = DST_OFF(n_off, c, 0, 0, 0, 0);
#else
        const int src_off = SRC_OFF(0, n_off, c, 0, 0, 0);
        const int dst_off = DST_OFF(0, n_off, c, 0, 0, 0);
#endif
        const vector_float src_vect = convert_vector_to_float(as_vector_data_t(
                sub_group_read((const __global BLOCK_DATA_T *)&src[src_off])));
        const vector_float diff_dst_vect
                = convert_vector_to_float(as_vector_data_t(sub_group_read(
                        (const __global BLOCK_DATA_T *)&diff_dst[dst_off])));

        diff_gamma_vect
                += (src_vect - mean_vect) * diff_dst_vect * inv_sqrt_variance;
        diff_beta_vect += diff_dst_vect;
    }

    float diff_gamma = 0, diff_beta = 0;
#if VECTOR_SIZE_SCALESHIFT == 1
    diff_gamma = diff_gamma_vect;
    diff_beta = diff_beta_vect;
#else
    for (int elem_idx = 0; elem_idx < VECTOR_SIZE_SCALESHIFT; elem_idx++) {
        diff_gamma += diff_gamma_vect[elem_idx];
        diff_beta += diff_beta_vect[elem_idx];
    }
#endif

    const int result_offset = n_chunk_idx * C + c;
    if (USE_SCALE || USE_SCALESHIFT)
        intel_sub_group_block_write((__global uint *)&diff_scale[result_offset],
                as_uint(diff_gamma));
    if (USE_SHIFT || USE_SCALESHIFT)
        intel_sub_group_block_write((__global uint *)&diff_shift[result_offset],
                as_uint(diff_beta));
}

NAMED_KERNEL_ATTR(SCALESHIFT_FINALIZE)
__kernel void ref_lnorm_bwd_scaleshift_final(__global float *tmp_reduce_mem,
        __global float *diff_scale, __global float *diff_shift) {
    const int c = GWS_GET_C_finalize();
    const int diff_shift_off = N_CHUNKS * C;
    __global float *tmp_diff_scale = tmp_reduce_mem;
    // diff_scale and diff_shift use the same tensor in scratchpad
    __global float *tmp_diff_shift = tmp_reduce_mem + diff_shift_off;

    float diff_gamma = 0;
    float diff_beta = 0;

    for (int n_chunk_idx = 0; n_chunk_idx < N_CHUNKS; n_chunk_idx++) {
        const int result_off = n_chunk_idx * C + c;
        diff_gamma += tmp_diff_scale[result_off];
        diff_beta += tmp_diff_shift[result_off];
    }

    if (diff_scale) diff_scale[c] = diff_gamma;
    if (diff_shift) diff_shift[SHIFT_OFF + c] = diff_beta;
}

#else // VECTORIZE_BWD_SCALESHIFT

NAMED_KERNEL_ATTR(SCALESHIFT)
__kernel void ref_lnorm_bwd_scaleshift(__global DATA_T *src,
        __global float *mean, __global float *variance,
        __global DATA_T *diff_dst, __global float *diff_scale,
        __global float *diff_shift, float eps) {

    const int c = GWS_GET_C();
    int x[6] = {0};

    float diff_gamma = 0;
    float diff_beta = 0;

    for (x[0] = 0; x[0] < max(1, STAT_D0); ++x[0]) {
        for (x[1] = 0; x[1] < max(1, STAT_D1); ++x[1]) {
            for (x[2] = 0; x[2] < max(1, STAT_D2); ++x[2]) {
                for (x[3] = 0; x[3] < max(1, STAT_D3); ++x[3]) {
                    x[NDIMS - 1] = 0;
                    const int s_off
                            = STAT_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

                    x[NDIMS - 1] = c;
                    const int src_off
                            = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
                    const int dst_off
                            = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

                    const float inv_sqrt_variance
                            = 1.0f / sqrt(variance[s_off] + eps);
                    const float dd = DST_TO_REF(diff_dst[dst_off]);

                    diff_gamma += (SRC_TO_REF(src[src_off]) - mean[s_off]) * dd
                            * inv_sqrt_variance;
                    diff_beta += dd;
                }
            }
        }
    }
    if (diff_scale) diff_scale[c] = diff_gamma;
    if (diff_shift) diff_shift[SHIFT_OFF + c] = diff_beta;
}
#endif // VECTORIZE_BWD_SCALESHIFT
#endif // USE_SCALESHIFT || USE_SCALE || USE_SHIFT

#if VECTORIZE_BWD

KERNEL_ATTR
__kernel void ref_lnorm_bwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *diff_dst,
        __global float *scale, __global DATA_T *diff_src, float eps) {
    int x[6] = {0};
    x[0] = GWS_GET_X0();
    x[1] = GWS_GET_X1();
    x[2] = GWS_GET_X2();
    x[3] = GWS_GET_X3();

    const int s_off = STAT_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
    const float mean_val = mean[s_off];
    const float inv_sqrt_variance = 1.0f / sqrt(variance[s_off] + eps);

    float dd_gamma = 0, dd_gamma_x = 0;
    VECT_FLOAT_T dd_gamma_vect = 0;
    VECT_FLOAT_T dd_gamma_x_vect = 0;
    if (CALCULATE_STATS) {
        for (int c = 0; c < C; c += VECT_DT_N * SUB_GROUP_SIZE) {
            VECT_FLOAT_T gamma = 1.0f;
            if (scale) {
                gamma = AS_VECT_FLOAT_T(
                        VECT_UINT_READ((const __global uint *)&scale[c]));
            }
            x[NDIMS - 1] = c;
            const int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
            const int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

            const VECT_FLOAT_T src_vect
                    = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                            (const __global BLOCK_DATA_T *)&src[src_off])));
            const VECT_FLOAT_T dst_vect
                    = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ((
                            const __global BLOCK_DATA_T *)&diff_dst[dst_off])));

            dd_gamma_vect += dst_vect * gamma;
            dd_gamma_x_vect += dst_vect * gamma * (src_vect - mean_val);
        }
#if VECT_DT_N == 1
        dd_gamma = dd_gamma_vect;
        dd_gamma_x = dd_gamma_x_vect;
#else
        for (int i = 0; i < VECT_DT_N; ++i) {
            dd_gamma += dd_gamma_vect[i];
            dd_gamma_x += dd_gamma_x_vect[i];
        }
#endif
        dd_gamma = sub_group_reduce_add(dd_gamma);
        dd_gamma_x = sub_group_reduce_add(dd_gamma_x);
        dd_gamma_x *= inv_sqrt_variance;
    }

    for (int c = 0; c < C; c += VECT_DT_N * SUB_GROUP_SIZE) {
        VECT_FLOAT_T gamma = 1.0f;
        if (scale) {
            gamma = AS_VECT_FLOAT_T(
                    VECT_UINT_READ((const __global uint *)&scale[c]));
        }
        x[NDIMS - 1] = c;
        const int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
        const int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

        const VECT_FLOAT_T src_vect = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(
                VECT_BLOCK_READ((const __global BLOCK_DATA_T *)&src[src_off])));
        VECT_FLOAT_T v_diff_src_vect
                = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                        (const __global BLOCK_DATA_T *)&diff_dst[dst_off])));
        v_diff_src_vect *= gamma;
        if (CALCULATE_STATS) {
            v_diff_src_vect -= dd_gamma / C
                    + (src_vect - mean_val) * dd_gamma_x * inv_sqrt_variance
                            / C;
        }
        v_diff_src_vect *= inv_sqrt_variance;
        VECT_BLOCK_WRITE((__global BLOCK_DATA_T *)&diff_src[src_off],
                AS_VECT_BLOCK_DATA_T(CONVERT_VECTOR_DATA_T(v_diff_src_vect)));
    }
}
#else // VECTORIZE_BWD
KERNEL_ATTR
__kernel void ref_lnorm_bwd(__global DATA_T *src, __global float *mean,
        __global float *variance, __global DATA_T *diff_dst,
        __global float *scale, __global DATA_T *diff_src, float eps) {

    int x[6] = {0};
    x[0] = GWS_GET_X0();
    x[1] = GWS_GET_X1();
    x[2] = GWS_GET_X2();
    x[3] = GWS_GET_X3();

    const int s_off = STAT_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
    const float mean_val = mean[s_off];

    const float inv_sqrt_variance = 1.0f / sqrt(variance[s_off] + eps);
    float dd_gamma = 0;
    float dd_gamma_x = 0;

    if (CALCULATE_STATS) {
        for (int c = 0; c < C; ++c) {
            const float gamma = scale ? scale[c] : 1.0f;

            x[NDIMS - 1] = c;
            const int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
            const int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

            const float dd = DST_TO_REF(diff_dst[dst_off]);
            dd_gamma += dd * gamma;
            dd_gamma_x += dd * gamma * (SRC_TO_REF(src[src_off]) - mean_val);
        }
        dd_gamma_x *= inv_sqrt_variance;
    }

    for (int c = 0; c < C; ++c) {
        const float gamma = scale ? scale[c] : 1.0f;

        x[NDIMS - 1] = c;
        const int src_off = SRC_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);
        const int dst_off = DST_OFF(x[0], x[1], x[2], x[3], x[4], x[5]);

        float v_diff_src = DST_TO_REF(diff_dst[dst_off]) * gamma;
        if (CALCULATE_STATS) {
            v_diff_src -= dd_gamma / C
                    + (SRC_TO_REF(src[src_off]) - mean_val) * dd_gamma_x
                            * inv_sqrt_variance / C;
        }
        v_diff_src *= inv_sqrt_variance;
        diff_src[src_off] = TO_SRC(v_diff_src);
    }
}
#endif // VECTORIZE_BWD
#endif // IS_BWD
