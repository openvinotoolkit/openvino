/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#if defined(IS_MAX)
#define INIT_ACC -INFINITY
#elif defined(IS_MIN)
#define INIT_ACC INFINITY
#elif defined(IS_MUL)
#define INIT_ACC 1.0f
#else
#define INIT_ACC 0.0f
#endif

#if defined(IS_MAX)
#define ACCUMULATE(x, y) fmax(x, y)
#elif defined(IS_MIN)
#define ACCUMULATE(x, y) fmin(x, y)
#elif defined(IS_MEAN) || defined(IS_SUM)
#define ACCUMULATE(x, y) (x + y)
#elif defined(IS_MUL)
#define ACCUMULATE(x, y) (x * y)
#else
#define ACCUMULATE(x, y) (x + pow(fabs(y), POWER))
#endif

// We want to use some acc algorithms (like pow) only once
// for a given element
#if defined(IS_MAX) || defined(IS_MIN) || defined(IS_MUL)
#define ACCUMULATE_AGAIN(x, y) ACCUMULATE(x, y)
#else
#define ACCUMULATE_AGAIN(x, y) (x + y)
#endif

#if defined(IS_MEAN)
#define FINALIZE(x) (x / REDUCTION_SIZE)
#elif defined(IS_LP_MAX)
#define FINALIZE(x) rootn(fmax(x, EPS), POWER)
#elif defined(IS_LP_SUM)
#define FINALIZE(x) rootn(x + EPS, POWER)
#elif defined(IS_P_MAX)
#define FINALIZE(x) fmax(x, EPS)
#elif defined(IS_P_SUM)
#define FINALIZE(x) (x + EPS)
#else
#define FINALIZE(x) (x)
#endif

#if defined(IS_MAX)
#define SUB_GROUP_REDUCE(x) sub_group_reduce_max(x)
#elif defined(IS_MIN)
#define SUB_GROUP_REDUCE(x) sub_group_reduce_min(x)
#elif defined(IS_MUL)
#define SUB_GROUP_REDUCE(x) \
    ({ \
        float sub_group_acc = 1.0; \
        for (int wi_id = 0; wi_id < SUB_GROUP_SIZE; wi_id++) { \
            sub_group_acc *= intel_sub_group_shuffle(c_acc, wi_id); \
        } \
        sub_group_acc; \
    })
#else
#define SUB_GROUP_REDUCE(x) sub_group_reduce_add(x)
#endif

#if INITIAL_C_CHUNKS == 1
#define C_BLOCK_READ BLOCK_READ
#define AS_C_BLOCK_DATA_T AS_DATA_T
#define CONVERT_C_BLOCK_FLOAT_T CONVERT_FLOAT_T
#define C_BLOCK_FLOAT_T float
#elif INITIAL_C_CHUNKS == 2
#define C_BLOCK_READ BLOCK_READ2
#define AS_C_BLOCK_DATA_T AS_DATA2_T
#define CONVERT_C_BLOCK_FLOAT_T CONVERT_FLOAT2_T
#define C_BLOCK_FLOAT_T float2
#endif

#define ROUND_DOWN(a, b) ((a) - ((a) % (b)))
#undef ROUND_UP
#define ROUND_UP(a, b) ROUND_DOWN((a + b - 1), (b))

// clang-format off
// C blocked or N,C blocked
#define INITIAL_SRC_OFFSET(n, c, hwd) \
    (((n) / N_BLOCK_SIZE) * INITIAL_HWD_DIM * N_BLOCK_SIZE * ROUND_UP(INITIAL_C, C_BLOCK_SIZE) + \
     ((n) % N_BLOCK_SIZE) * C_BLOCK_SIZE + \
     ((c) / C_BLOCK_SIZE) * INITIAL_HWD_DIM * N_BLOCK_SIZE * C_BLOCK_SIZE + \
     (hwd) * N_BLOCK_SIZE * C_BLOCK_SIZE + \
     ((c) % C_BLOCK_SIZE))

#define INITIAL_DST_OFFSET(n, c, hwd) \
    ((n / N_BLOCK_SIZE) * FINAL_HWD_DIM * N_BLOCK_SIZE * ROUND_UP(FINAL_C_DIM, C_BLOCK_SIZE) + \
     ((n) % N_BLOCK_SIZE) * C_BLOCK_SIZE + \
     ((c) / C_BLOCK_SIZE) * FINAL_HWD_DIM * N_BLOCK_SIZE * C_BLOCK_SIZE + \
     (hwd) * N_BLOCK_SIZE * C_BLOCK_SIZE + \
     ((c) % C_BLOCK_SIZE))

#define FINAL_SRC_OFFSET(n, c, hwd) INITIAL_DST_OFFSET(n, c, hwd)

#define FINAL_DST_OFFSET(n, c, hwd) \
    ((n) / N_BLOCK_SIZE) * (FINAL_HWD_DIM / FINAL_HWD_CHUNK_SIZE) * N_BLOCK_SIZE * ROUND_UP(FINAL_C_DIM / FINAL_C_CHUNK_SIZE, C_BLOCK_SIZE) + \
     ((n) % N_BLOCK_SIZE) * C_BLOCK_SIZE + \
     ((c) / C_BLOCK_SIZE) * (FINAL_HWD_DIM / FINAL_HWD_CHUNK_SIZE) * N_BLOCK_SIZE * C_BLOCK_SIZE + \
     (hwd) * N_BLOCK_SIZE * C_BLOCK_SIZE + \
     ((c) % C_BLOCK_SIZE)
// clang-format on

#if SKIP_FINAL_PHASE
#define WRITE_INITIAL_RESULT(dst_ptr, dst_offset, data) \
    { dst_ptr[dst_offset] = TO_DST(FINALIZE(data)); }
#define INITIAL_DST_DTYPE DST_DATA_T
#else
#define WRITE_INITIAL_RESULT(dst_ptr, dst_offset, data) \
    { dst_ptr[dst_offset] = data; }
#define INITIAL_DST_DTYPE float
#endif

// Reduces only chunks of reduction dimensions
// in order to create more threads and increase precision
NAMED_KERNEL_ATTR(INITIAL)
__kernel void gen9_initial_reduce(
        __global SRC_DATA_T *src, __global INITIAL_DST_DTYPE *dst) {
    const int n_chunk_idx = GWS_GET_INITIAL_N();
    const int c = GWS_GET_INITIAL_C();
    const int c_block_idx = c / C_BLOCK_SIZE;
    const int hwd_chunk_idx = GWS_GET_INITIAL_HWD_CHUNK_ID();

    const int hwd_start = hwd_chunk_idx * INITIAL_HWD_CHUNK_SIZE;
    const int current_hwd_chunk = min(INITIAL_HWD_CHUNK_SIZE,
            INITIAL_HWD_DIM - hwd_chunk_idx * INITIAL_HWD_CHUNK_SIZE);
    const int aligned_hwd_chunk = ROUND_DOWN(current_hwd_chunk, VECT_DT_N);

    const int n_start = n_chunk_idx * INITIAL_N_CHUNK_SIZE;
    const int n_end = min(n_start + INITIAL_N_CHUNK_SIZE, INITIAL_N);

#if SKIP_FINAL_PHASE
    // zero pad dst memory
    for (int n_idx = n_start; n_idx < n_start + INITIAL_N_CHUNK_SIZE; n_idx++) {
        for (int c_idx = c; c_idx < c + INITIAL_C_CHUNKS * SUB_GROUP_SIZE;
                c_idx++) {
            if (n_idx >= DST_N && n_idx < DST_N_PADDED
                    || c_idx >= DST_C && c_idx < DST_C_PADDED) {
                for (int hwd_idx = hwd_start;
                        hwd_idx < hwd_start + INITIAL_HWD_CHUNK_SIZE;
                        hwd_idx++) {
                    const int dst_off = FINAL_DST_OFFSET(n_idx, c_idx, hwd_idx);
                    dst[dst_off] = TO_DST(0.0f);
                }
            }
        }
    }
#endif
    if (c >= INITIAL_C || n_start >= INITIAL_N) { return; }

    VECT_FLOAT_T vector_acc = INIT_ACC;
    for (int n = n_start; n < n_end; n++) {
        for (int hwd_id = 0; hwd_id < aligned_hwd_chunk; hwd_id += VECT_DT_N) {
            for (int c_chunk = 0; c_chunk < INITIAL_C_CHUNKS; c_chunk++) {
                // It will always read from the beginning of c block
                const int off = INITIAL_SRC_OFFSET(n, c,
                        hwd_start + hwd_id
                                + c_chunk * VECT_DT_N / INITIAL_C_CHUNKS);
                VECT_FLOAT_T data
                        = CONVERT_VECT_FLOAT_T(AS_VECT_DATA_T(VECT_BLOCK_READ(
                                (const __global BLOCK_DATA_T *)&src[off])));
                vector_acc = ACCUMULATE(vector_acc, data);
            }
        }
        for (int hwd_id = aligned_hwd_chunk; hwd_id < current_hwd_chunk;
                hwd_id++) {
            const int off = INITIAL_SRC_OFFSET(n, c, hwd_start + hwd_id);
            C_BLOCK_FLOAT_T data = CONVERT_C_BLOCK_FLOAT_T(AS_C_BLOCK_DATA_T(
                    C_BLOCK_READ((const __global BLOCK_DATA_T *)&src[off])));
#if VECT_DT_N == 1
            vector_acc = ACCUMULATE(vector_acc, data);
#else // VECT_DT_N == 1
#if INITIAL_C_CHUNKS == 1
            vector_acc[0] = ACCUMULATE(vector_acc[0], data);
#elif INITIAL_C_CHUNKS == 2
            // data[0] and data[1] must be accumulated separately as they contain different C
            vector_acc[0] = ACCUMULATE(vector_acc[0], data[0]);
            vector_acc[1] = ACCUMULATE(vector_acc[1], data[1]);
#endif // INITIAL_C_CHUNKS == 1
#endif // VECT_DT_N == 1
        }
    }
#if VECT_DT_N == 1
    float acc = vector_acc;
#else // VECT_DT_N == 1
    const int elems_to_accumulate = aligned_hwd_chunk > 0 ? VECT_DT_N : 1;
#if INITIAL_C_CHUNKS == 1
    float acc = INIT_ACC;
    for (int i = 0; i < elems_to_accumulate; i++) {
        acc = ACCUMULATE_AGAIN(acc, vector_acc[i]);
    }
#elif INITIAL_C_CHUNKS == 2
    float2 acc = INIT_ACC;
    for (int i = 0; i < elems_to_accumulate; i += 2) {
        acc[0] = ACCUMULATE_AGAIN(acc[0], vector_acc[i]);
        acc[1] = ACCUMULATE_AGAIN(acc[1], vector_acc[i + 1]);
    }
#endif // INITIAL_C_CHUNKS == 1
#endif // VECT_DT_N == 1

    const int local_id = get_sub_group_local_id();
#if IS_C_REDUCED
#if INITIAL_C_CHUNKS == 2
    float c_acc = acc[0] + acc[1];
#elif INITIAL_C_CHUNKS == 1
    float c_acc = acc;
#endif // INITIAL_C_CHUNKS == 2
    const int dst_off
            = INITIAL_DST_OFFSET(n_chunk_idx, c_block_idx, hwd_chunk_idx);
    c_acc = SUB_GROUP_REDUCE(c_acc);
    if (local_id == 0) { WRITE_INITIAL_RESULT(dst, dst_off, c_acc); }
#else // IS_C_REDUCED
    const int dst_c = c + local_id;
#if INITIAL_C_CHUNKS == 1
    WRITE_INITIAL_RESULT(
            dst, INITIAL_DST_OFFSET(n_chunk_idx, dst_c, hwd_chunk_idx), acc);
#else // INITIAL_C_CHUNKS == 1
    WRITE_INITIAL_RESULT(
            dst, INITIAL_DST_OFFSET(n_chunk_idx, dst_c, hwd_chunk_idx), acc[0]);
    WRITE_INITIAL_RESULT(dst,
            INITIAL_DST_OFFSET(
                    n_chunk_idx, dst_c + SUB_GROUP_SIZE, hwd_chunk_idx),
            acc[1]);
#endif // INITIAL_C_CHUNKS == 1
#endif // IS_C_REDUCED
}

// Finalizes reduction by reducing results of initial reduction
NAMED_KERNEL_ATTR(FINAL)
__kernel void gen9_final_reduce(__global float *src, __global DST_DATA_T *dst) {
    const int n_start = GWS_GET_FINAL_N() * FINAL_N_CHUNK_SIZE;
    const int c_start = GWS_GET_FINAL_C() * FINAL_C_CHUNK_SIZE;
    const int hwd_start = GWS_GET_FINAL_HWD() * FINAL_HWD_CHUNK_SIZE;

    float acc = INIT_ACC;
    const int max_n = max(DST_N_PADDED, FINAL_N_DIM);
    const int max_c = max(DST_C_PADDED, FINAL_C_DIM);
    const int n_end = min(max_n, n_start + FINAL_N_CHUNK_SIZE);
    const int c_end = min(max_c, c_start + FINAL_C_CHUNK_SIZE);
    const int hwd_end = min(FINAL_HWD_DIM, hwd_start + FINAL_HWD_CHUNK_SIZE);
    for (int n = n_start; n < n_end; n++) {
        for (int c = c_start; c < c_end; c++) {
            for (int hwd = hwd_start; hwd < hwd_end; hwd++) {
                // zero pad dst memory
                if ((n >= DST_N && n < DST_N_PADDED)
                        || (c >= DST_C && c < DST_C_PADDED)) {
                    const int dst_off = FINAL_DST_OFFSET(n, c, hwd);
                    dst[dst_off] = TO_DST(0.0f);
                }
                if (n < FINAL_N_DIM && c < FINAL_C_DIM) {
                    const int off = FINAL_SRC_OFFSET(n, c, hwd);
                    const float data = src[off];
                    acc = ACCUMULATE_AGAIN(acc, data);
                }
            }
        }
    }
    if (n_start < DST_N && c_start < DST_C) {
        const int off = FINAL_DST_OFFSET(n_start, c_start, hwd_start);
        dst[off] = TO_DST(FINALIZE(acc));
    }
}
