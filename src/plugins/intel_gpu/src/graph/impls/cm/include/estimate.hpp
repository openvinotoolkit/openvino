/*
 * Copyright (c) 2020-2023, Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <cm/cm.h>
#include <cm/cmtl.h>

#if defined(SHIM) || defined(CMRT_EMU)
#define ATTR
#define ATTR_BUF
#define CM_LOCAL_BARRIER 0x20
#include "emu/block2d.h"
#else
#define ATTR [[type("svmptr_t")]]
#define ATTR_BUF [[type("buffer_t")]]
#endif

#define MYMIN(x, y) ((x) < (y) ? (x) : (y))

template<typename T, int N>
void show(const vector<T, N> mat) {
    printf("vector [%d]:\n[", N);
    for(int n = 0; n < N; n ++) {
        printf("%8.4f,", mat[n]);
    }
    printf("]\n");
}

template<typename T, int N>
void show_i(const vector<T, N> mat) {
    printf("vector [%d]:\n[", N);
    for(int n = 0; n < N; n ++) {
        printf("%d,", mat[n]);
    }
    printf("]\n");
}

template<typename T, int M, int N>
void show(const matrix<T, M, N> mat) {
    printf("Matrix [%d, %d]:\n", M, N);
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            printf("%8.4f,", mat[m][n]);
        }
        printf("],\n");
    }
    printf("]\n");
}

template<typename T, int M, int N>
void show(const matrix_ref<T, M, N> mat) {
    printf("Matrix [%d, %d]:\n", M, N);
    for(int m = 0; m < M; m ++) {
        printf("\t[");
        for(int n = 0; n < N; n ++) {
            printf("%8.4f,", mat[m][n]);
        }
        printf("],\n");
    }
    printf("]\n");
}

// group_count: M = group_count * group_size, group_size is the element count of current reduction
// op: 0-max, 1-sum
// M: before reduce element count, N: row count, stop: element count M must be larger than
template <int group_count, int op, int stop, typename T, int N, int M>
CM_INLINE constexpr auto reduce2d(matrix_ref<T, N, M> src) {
    constexpr int group_size = M / group_count;
    if constexpr (N > stop && group_size > 1) {
        matrix<T, N / 2, M> result;
        // half of group will be reduced
        constexpr int new_group_size = group_size / 2;
        constexpr int new_group_count = group_count * 2;
#pragma unroll
        for (int i = 0; i < N / 2; i++) {
            matrix<T, group_count * 2, new_group_size> new_top, new_bot;
            auto top = src.row(2 * i + 0).format<T, new_group_count, new_group_size>();
            auto bot = src.row(2 * i + 1).format<T, new_group_count, new_group_size>();
            constexpr int v_stride = new_group_count == 2 ? 1 : 2;

            new_top.select<new_group_count / 2, 1, new_group_size, 1>(0) = top.select<new_group_count / 2, v_stride, new_group_size, 1>(0);
            new_top.select<new_group_count / 2, 1, new_group_size, 1>(new_group_count / 2) = bot.select<new_group_count / 2, v_stride, new_group_size, 1>(0);
            new_bot.select<new_group_count / 2, 1, new_group_size, 1>(0) = top.select<new_group_count / 2, v_stride, new_group_size, 1>(1);
            new_bot.select<new_group_count / 2, 1, new_group_size, 1>(new_group_count / 2) = bot.select<new_group_count / 2, v_stride, new_group_size, 1>(1);
            if constexpr (op == 0) {
                result[i] = cm_max<T>(new_top.format<T>(), new_bot.format<T>());
            } else {
                result[i] = (new_top.format<T>() + new_bot.format<T>());
            }
        }

        return reduce2d<group_count * 2, op, stop>(result);
    } else {
        matrix<T, N, M> dst = src;
        return dst;
    }
}

template <typename T, int N>
CM_INLINE void read_1d(vector_ref<T, N> out, svmptr_t base) {
    cm_ptr_block_read((T*)base, out);
}

template <typename T, int M, int N>
CM_INLINE void read_2d(matrix_ref<T, M, N> out, svmptr_t base, uint pitch) {
#pragma unroll
    for (int i = 0; i < out.n_rows(); i++, base += pitch) {
        cm_ptr_block_read((T*)base, out.row(i));
    }
}

template <typename TSRC, int M, int N>
CM_INLINE void write_2d(matrix_ref<TSRC, M, N> out, svmptr_t base, uint pitch) {
#pragma unroll
    for (int i = 0; i < out.n_rows(); i++, base += pitch) {
        cm_ptr_block_write((TSRC*)base, out.row(i));
    }
}

template <typename TSRC, int M, int N>
CM_INLINE void write_2d(matrix_ref<TSRC, M, N> out, SurfaceIndex base, uint offset, uint pitch) {
#pragma unroll
    for (int i = 0; i < out.n_rows(); i++, offset += pitch) {
        cm_store<int, N / (sizeof(int) / sizeof(TSRC)), DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(base, offset, out.row(i).format<int>());
    }
}

template <int M, int N>
CM_INLINE void cm_load_2d(matrix_ref<uint, M, N> out, SurfaceIndex base, uint offset, uint pitch) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        out.row(i).format<uint>() = cm_load<uint, N>(base, offset + i * pitch);
    }
}

template <int M, int N>
CM_INLINE void cm_load_2d(matrix_ref<half, M, N> out, SurfaceIndex base, uint offset, uint pitch, bool showit = 0) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        out.row(i).format<uint>() = cm_load<uint, N/2,DataSize::U32, CacheHint::Cached, CacheHint::Cached>(base, offset + i * pitch);
    }
}

//half
template <int M, int N>
CM_INLINE void cm_prefetch_2d(SurfaceIndex base, uint offset, uint pitch) {
    #pragma unroll
    for(int i = 0; i < M; i++) {
        cm_prefetch<N/2, DataSize::U32, CacheHint::Cached, CacheHint::Cached>(base, offset + i * pitch);
    }
}

template <int M, int N>
CM_INLINE void cm_gather_2d(matrix_ref<half, M, N> out, SurfaceIndex base, uint offset, uint pitch, bool showit = 0) {
    vector<uint, M> offsets;
    #pragma unroll
    for(int i = 0; i < M; i++) {
        offsets[i] = offset + i * pitch;
    }
    // if (showit) {
    //     // printf("M=%d", M);
    //     show_i(offsets);
    // }
    out.format<uint>() = cm_load<uint, VectorSize::N8>(base, offsets);
    // if (showit) {
    //     show(out.format<half, M, N>());
    // }
}

template <int M, int N>
CM_INLINE void cm_store_2d(matrix_ref<half, M, N> out, SurfaceIndex base, uint offset, uint pitch) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        cm_store<uint, N/2>(base, offset + i * pitch, out.row(i).format<uint>());
    }
}

template <int M, int N>
CM_INLINE void cm_store_2d(matrix_ref<uint, M, N> out, SurfaceIndex base, uint offset, uint pitch) {
    #pragma unroll
    for(int i = 0; i < out.n_rows(); i++) {
        cm_store<uint, N>(base, offset + i * pitch, out.row(i).format<uint>());
    }
}

#if USE_INT8
    #define KV_ELEMENT_TYPE uint8_t
#else
    #define KV_ELEMENT_TYPE half
#endif

#if 1 || (BLOCK_SG_M == 64 && BLOCK_SG_N == 32)
// const static int channels_reduce_32[] = { 0, 16,  8, 24,  4, 20, 12, 28,  2, 18, 10, 26,  6, 22, 14, 30,  
// 		                                  1, 17,  9, 25,  5, 21, 13, 29,  3, 19, 11, 27,  7, 23, 15, 31};
// src_a is query, src_b is key

CM_INLINE void gemm_qk(uint id_wg_m, uint id_wg_n, uint hq, uint slm,
        #ifdef CM_HAS_LSC_UNTYPED_2D
        svmptr_t key_cache ATTR,
        svmptr_t query ATTR,
        #else
        SurfaceIndex key_cache [[type("buffer_t")]],
        SurfaceIndex query [[type("buffer_t")]],
        #endif
        svmptr_t block_indices ATTR,
        svmptr_t block_indices_begins ATTR,
        svmptr_t kq_max_wg ATTR,
        #ifdef CM_HAS_LSC_UNTYPED_2D
        svmptr_t kq_exp_partial_sum ATTR,
        #else
        SurfaceIndex kq_exp_partial_sum [[type("buffer_t")]],
        #endif
        const uint M, const uint N, const uint K, const uint query_stride, const uint q_start_strided, const uint offset_partial_sum) {

    constexpr int SG_SIZE = details::get_dpas_execution_size((CmPrecisionType)9);    
    // constexpr int BLOCK_WG_K = 64;	// same in sg  // because unroll 4 times along K ??
    constexpr int SUM_N = BLOCK_SG_N / (BLOCK_SIZE/STRIDE);
    // xehpg DPAS spec: dst: [8, 8], repeat: 1~8, depth: 8
    static constexpr int REPEAT = 8;
    static constexpr int DEPTH = 8;
    static constexpr int BLOCK_REG_M = REPEAT;      // 8
    static constexpr int BLOCK_REG_N = SG_SIZE;     // 16 Xe1?
    static constexpr int BLOCK_DPAS_C = BLOCK_REG_M * BLOCK_REG_N;  // src0
    static constexpr int VNNI = sizeof(half);
    static constexpr int BLOCK_REG_K = DEPTH * sizeof(int) / VNNI;   // 8*2
    static constexpr int BLOCK_REG_A = BLOCK_REG_M * BLOCK_REG_K;    // src2
    static constexpr int BLOCK_REG_B = BLOCK_REG_N * BLOCK_REG_K;    // scr1

    static constexpr int BLOCK_WG_M = SG_M * BLOCK_SG_M;   // 4*64=256
    static constexpr int BLOCK_WG_N = SG_N * BLOCK_SG_N;   // 8*32=256
    // register blocking
    static constexpr int REG_M = BLOCK_SG_M / BLOCK_REG_M;  // 64/8=8 times per thread
    static constexpr int REG_N = BLOCK_SG_N / BLOCK_REG_N;  // 32/16=2 times per thread Xe1?
    static constexpr int REG_K = BLOCK_WG_K / BLOCK_REG_K;  // 64/16=4
    static constexpr int REG_MN = REG_M * REG_N;
    #ifdef CM_HAS_LSC_UNTYPED_2D
    static constexpr int KEY_LINES_PER_LOAD = KV_BLOCK_SIZE / STRIDE;   // 256/16=16, i.e. BLOCK_REG_N because KV_BLOCK_SIZE=BLOCK_REG_N*STRIDE ?
    #else
    static constexpr int KEY_LINES_PER_LOAD = SG_SIZE;
    vector<uint, KEY_LINES_PER_LOAD> gather_offsets_b;
    cmtl::cm_vector_assign(gather_offsets_b.select_all(), 0, 1);
    vector<uint, BLOCK_SG_M> block_offsets_a;
    cmtl::cm_vector_assign(block_offsets_a.select_all(), 0, 1);
    #endif 
    matrix<float, REG_M * REG_N, BLOCK_DPAS_C> acc = 0;                              // --> 64*2 regs   // 8*2 x (8*16) Xe1?
    uint id_sg_n = cm_local_id(0);
    uint id_sg_m = cm_local_id(1);
    uint id_sg_mn = id_sg_m * SG_N + id_sg_n;

    static_assert(BLOCK_WG_K == 32 || BLOCK_WG_K == 64, "Supported BLOCK_WG_K are 32 or 64");
    constexpr bool IS64 = (BLOCK_WG_K == 64);
    constexpr bool IS32 = (BLOCK_WG_K == 32);
    constexpr int PHASE_SIZE = IS64 ? 32 : 16;

    static_assert(REG_N == 2, "block_2d_desc for b is manually unrolled by 2");
    static_assert(HEAD_SIZE % BLOCK_WG_K == 0, "K dimension must be multiple of BLOCK_WG_K");
    static_assert(KV_BLOCK_SIZE == 256, "block size of key(key_cache) should be 256");
    uint N_block = (N + BLOCK_WG_N - 1) / BLOCK_WG_N;
    uint M_aligned = (M + BLOCK_WG_M - 1) / BLOCK_WG_M * BLOCK_WG_M;
    uint K_block_pad = N_block * (BLOCK_WG_N / (BLOCK_SIZE / STRIDE));
    const uint block_size_div_stride = BLOCK_SIZE / STRIDE;  // 8
    constexpr SOFTMAX_TYPE log2e = 1.4426950408889634f;
    constexpr uint k_block_in_group = BLOCK_WG_N / block_size_div_stride;  // 32
    constexpr uint k_block_in_sg = k_block_in_group / SG_N;   // 4

    static constexpr int SG_MN = SG_M * SG_N;

#if IS_CAUSAL == 1
    if ((int)(id_wg_n * BLOCK_WG_N) >= ((int)id_wg_m + 1) * BLOCK_WG_M + q_start_strided) {
        // fill -inf -> max in group, 0 -> exp_sum to make compensation work
        {
            // current max -> mem
            vector<SOFTMAX_TYPE, BLOCK_SG_M> max_m = -60000;
            // kq_max_wg: [b, hq, N/BLOCK_WG_N, M_aligned]
            uint offset = (id_wg_n * M_aligned + id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) * sizeof(SOFTMAX_TYPE);
            cm_ptr_store<int>((int*)kq_max_wg, offset, max_m.format<int>());
        }
        {
            // store
            #ifdef CM_HAS_LSC_UNTYPED_2D
            matrix<SOFTMAX_TYPE, 8, 4> sum_t = 0;
            lsc::block_2d_desc<SOFTMAX_TYPE, 1, 8, 4> desc_c{ kq_exp_partial_sum, M - 1, (uint)(K_block_pad * sizeof(SOFTMAX_TYPE) - 1), (uint)(K_block_pad * sizeof(SOFTMAX_TYPE) - 1),
                (int)((id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) / block_size_div_stride), (int)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) };
            cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 0>(desc_c, sum_t.format<SOFTMAX_TYPE>());
            cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 1>(desc_c, sum_t.format<SOFTMAX_TYPE>());
            cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 2>(desc_c, sum_t.format<SOFTMAX_TYPE>());
            cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 3>(desc_c, sum_t.format<SOFTMAX_TYPE>());
            cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 4>(desc_c, sum_t.format<SOFTMAX_TYPE>());
            cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 5>(desc_c, sum_t.format<SOFTMAX_TYPE>());
            cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 6>(desc_c, sum_t.format<SOFTMAX_TYPE>());
            cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 7>(desc_c, sum_t.format<SOFTMAX_TYPE>());
            #else
            matrix<SOFTMAX_TYPE, 8, SUM_N> sum_t = 0;
            const uint pitch_c = K_block_pad * sizeof(SOFTMAX_TYPE);
            uint off_c = offset_partial_sum;
            off_c += pitch_c * (id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M);
            off_c += ((id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) / block_size_div_stride) * sizeof(SOFTMAX_TYPE);

            #pragma unroll
            for (uint i = 0; i < BLOCK_SG_M/8; i++) {
                cm_store_2d(sum_t.format<uint, 8, SUM_N>(), kq_exp_partial_sum, off_c + (8 * i) * pitch_c, pitch_c);
            }
            #endif

        }

        return;
    }
#endif
    // assume block index coming from 0 in block_indices_begins
    int block_index_begin = ((int*)block_indices_begins)[0];
    int* block_indices_p = (int*)block_indices + block_index_begin;
    int b_adjacent_between_head = query_stride / STRIDE;   // HEAD_SIZE * HQ  +  HEAD_SIZE * HQ (padding 0)
    // M[0:16*2]xK[0:16]
    #ifdef CM_HAS_LSC_UNTYPED_2D
    lsc::block_2d_desc<half, 1, 32, BLOCK_REG_K> desc_a{ query, M - 1, (uint)((query_stride - hq * HEAD_SIZE) * sizeof(half) - 1), (uint)(query_stride * sizeof(half) - 1),
        (STRIDE - 1) * b_adjacent_between_head, (int)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) };   // 32 = BLOCK_SG_M/2 ?
    // prefetch A
    lsc::block_2d_desc<half, 1, BLOCK_WG_M / SG_MN, PHASE_SIZE> desc_prefetch_a{query, M - 1, (uint)((query_stride - hq * HEAD_SIZE) * sizeof(half) - 1), (uint)(query_stride * sizeof(half) - 1),
        (STRIDE - 1) * b_adjacent_between_head, (int)(id_wg_m * BLOCK_WG_M + id_sg_mn * (BLOCK_WG_M / SG_MN)) };
    #else
    const uint pitch_a = query_stride * sizeof(half);
    uint base_off_a = hq * HEAD_SIZE * (uint)sizeof(half);  /*q header base*/
    uint base_off_prefetch_a = base_off_a;

    base_off_a += pitch_a * /*Y_offset*/(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M);
    uint off_a = base_off_a + /*X_offset*/ ((STRIDE - 1) * b_adjacent_between_head) * sizeof(half);

    base_off_prefetch_a += pitch_a * (id_wg_m * BLOCK_WG_M + id_sg_mn * (BLOCK_WG_M / SG_MN));
    uint off_prefetch_a = base_off_prefetch_a + ((STRIDE - 1) * b_adjacent_between_head) * sizeof(half);
    #endif
    // M[0:16*2]xK[0:16]    // should be M[0:32*2]xK[0:16] instead ?                                                              --> 8+8 regs
    matrix<half, REG_M, BLOCK_REG_A> a0;

    // M[0:16]xK[0:32]
    uint block_idx = (uint)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) * STRIDE / KV_BLOCK_SIZE;
    uint max_block_idx = (uint)(N * STRIDE + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE - 1;
    block_idx = MYMIN(block_idx, max_block_idx);
#if USE_INT8
    uint offset = block_indices_p[block_idx] * (HK * KV_BLOCK_SIZE * HEAD_SIZE_KEY * (uint)sizeof(char));
    #ifdef CM_HAS_LSC_UNTYPED_2D
    lsc::block_2d_desc<int, 1, KEY_LINES_PER_LOAD, 8> desc_b0{ key_cache + offset, KEY_LINES_PER_LOAD - 1, (uint)(K * sizeof(char) - 1), (uint)(K * sizeof(char) - 1),
        0, 0 };
    uint scale_offset0 = offset + KV_BLOCK_SIZE * HEAD_SIZE;   //KV_BLOCK_SIZE *128(int8)  + KV_BLOCK_SIZE(half)   + KV_BLOCK_SIZE(half)
    #else 
    const uint pitch_b = K * sizeof(char);
    uint hk = hq / (HQ / HK);
    uint off_b0 = hk * (KV_BLOCK_SIZE * HEAD_SIZE_KEY * (uint)sizeof(char)) + offset;
    uint scale_offset0 = off_b0 + KV_BLOCK_SIZE * HEAD_SIZE;   //KV_BLOCK_SIZE *128(int8)  + KV_BLOCK_SIZE(half)   + KV_BLOCK_SIZE(half) 
    #endif 

    block_idx = MYMIN(block_idx + 1, max_block_idx);
    offset = block_indices_p[block_idx] * (HK * KV_BLOCK_SIZE * HEAD_SIZE_KEY * (uint)sizeof(char));
    #ifdef CM_HAS_LSC_UNTYPED_2D
    lsc::block_2d_desc<int, 1, KEY_LINES_PER_LOAD, 8> desc_b1{ key_cache + offset, KEY_LINES_PER_LOAD - 1, (uint)(K * sizeof(char) - 1), (uint)(K * sizeof(char) - 1),
        0, 0 };
    uint scale_offset1 = offset + KV_BLOCK_SIZE * HEAD_SIZE;
    #else
    uint off_b1 = off_b0 + KEY_LINES_PER_LOAD * pitch_b; 
    //uint off_b1 = hk * (KV_BLOCK_SIZE * HEAD_SIZE_KEY * (uint)sizeof(half)) + offset;
    uint scale_offset1 = scale_offset0 + KV_BLOCK_SIZE * 2;
    //off_b1 += offset;
    #endif 

    // prefetch B
    block_idx = (uint)(id_wg_n * BLOCK_WG_N + id_sg_mn * (BLOCK_WG_N / SG_MN)) * STRIDE / KV_BLOCK_SIZE;
    block_idx = MYMIN(block_idx, max_block_idx);
    offset = block_indices_p[block_idx] * (HK * KV_BLOCK_SIZE * HEAD_SIZE_KEY * (uint)sizeof(char));
    static_assert(BLOCK_WG_N / SG_MN <= KEY_LINES_PER_LOAD, "prefetch lines should be inside one block");
    #ifdef CM_HAS_LSC_UNTYPED_2D
    lsc::block_2d_desc<uchar, 1, BLOCK_WG_N / SG_MN, PHASE_SIZE> desc_prefetch_b{ key_cache + offset, BLOCK_WG_N / SG_MN - 1, (uint)(K * sizeof(char) - 1), (uint)(K * sizeof(char) - 1),
        0, 0 };
    #else
    uint off_prefetch_b = hk * (KV_BLOCK_SIZE * HEAD_SIZE_KEY * (uint)sizeof(char));
    off_prefetch_b += offset;
    vector<uint, KEY_LINES_PER_LOAD> offsets_scale0 = (gather_offsets_b) * STRIDE * sizeof(half) + scale_offset0;
    vector<uint, KEY_LINES_PER_LOAD> offsets_scale1 = offsets_scale0 + KV_BLOCK_SIZE*2;
    #endif

    // N[:]xK[0:32]                                                     --> 16 * 1 regs
    matrix<int, KEY_LINES_PER_LOAD, 8> b0_up_s8, b0_down_s8, b1_up_s8, b1_down_s8; //ping pong
    matrix<half, REG_N, BLOCK_REG_B> b0;                      // after dequant
    matrix<half, REG_N, KEY_LINES_PER_LOAD * 2> scales, zps;
    matrix<half, REG_N, KEY_LINES_PER_LOAD*STRIDE> scales_block, zps_block;
#else
    uint offset = block_indices_p[block_idx] * (HK * KV_BLOCK_SIZE * HEAD_SIZE * (uint)sizeof(half));
    #ifdef CM_HAS_LSC_UNTYPED_2D
    lsc::block_2d_desc<int, 1, KEY_LINES_PER_LOAD, 8> desc_b0{ key_cache + offset, KEY_LINES_PER_LOAD - 1, (uint)(K * sizeof(half) - 1), (uint)(K * sizeof(half) - 1),
        0, 0 };
    #else
    const uint pitch_b = K * sizeof(half);
    uint hk = hq / (HQ / HK);
    uint off_b0 = hk * (KV_BLOCK_SIZE * HEAD_SIZE_KEY * (uint)sizeof(half)) + offset;
    //off_b0 += offset;
    #endif
    block_idx = MYMIN(block_idx + 1, max_block_idx);
    offset = block_indices_p[block_idx] * (HK * KV_BLOCK_SIZE * HEAD_SIZE * (uint)sizeof(half));
    #ifdef CM_HAS_LSC_UNTYPED_2D
    lsc::block_2d_desc<int, 1, KEY_LINES_PER_LOAD, 8> desc_b1{ key_cache + offset, KEY_LINES_PER_LOAD - 1, (uint)(K * sizeof(half) - 1), (uint)(K * sizeof(half) - 1),
        0, 0 };
    #else
    //uint off_b1 = hk * (KV_BLOCK_SIZE * HEAD_SIZE_KEY * (uint)sizeof(half)) + offset;
    uint off_b1 = off_b0 + KEY_LINES_PER_LOAD * pitch_b;
    //off_b1 += offset;
    #endif
    // prefetch B
    block_idx = (uint)(id_wg_n * BLOCK_WG_N + id_sg_mn * (BLOCK_WG_N / SG_MN)) * STRIDE / KV_BLOCK_SIZE;
    block_idx = MYMIN(block_idx, max_block_idx);
    offset = block_indices_p[block_idx] * (HK * KV_BLOCK_SIZE * HEAD_SIZE * (uint)sizeof(half));  //paged offset
    static_assert(BLOCK_WG_N / SG_MN <= KEY_LINES_PER_LOAD, "prefetch lines should be inside one block");
    #ifdef CM_HAS_LSC_UNTYPED_2D
    lsc::block_2d_desc<half, 1, BLOCK_WG_N / SG_MN, PHASE_SIZE> desc_prefetch_b{ key_cache + offset, BLOCK_WG_N / SG_MN - 1, (uint)(K * sizeof(half) - 1), (uint)(K * sizeof(half) - 1),
        0, 0 };
    #else
    uint off_prefetch_b = hk * (KV_BLOCK_SIZE * HEAD_SIZE_KEY * (uint)sizeof(half));
    off_prefetch_b += offset;

    #endif
    // 0~2 M[:]xK[0:16] 2~4 K[16:32]                                                     --> 32 * 2 regs
    matrix<half, REG_N, BLOCK_REG_B> b0, b1;      // ping-pong B
#endif

    if constexpr (IS64) {
        // warmup
        // prefetch
        #ifdef CM_HAS_LSC_UNTYPED_2D
        cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
        desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);
        cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
        desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);
        #else
        cm_prefetch_2d<BLOCK_WG_N / SG_MN, 32>(key_cache,off_prefetch_b,pitch_b);
        off_prefetch_b += 32 * sizeof(half);  
        cm_prefetch_2d<BLOCK_WG_M / SG_MN, 32>(query,off_prefetch_a,pitch_a);
        off_prefetch_a += 32 * sizeof(half);
        #endif
    }

    // load b: N[0:16]xK[0:16]
#if USE_INT8
    {
        #ifdef CM_HAS_LSC_UNTYPED_2D
        //16 * 16 half,  1D->2D
        matrix<half, KEY_LINES_PER_LOAD, STRIDE> tmp_scale, tmp_zp;  //KV_BLOCK_SIZE/2
        lsc::block_2d_desc<int, 1, 16, 16 / 2> desc_scale{ key_cache + scale_offset0, KV_BLOCK_SIZE/16 * 2 - 1, (uint)(16 * sizeof(half) - 1), (uint)(16 * sizeof(half) - 1),
            0, 0 };
        cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 0>(tmp_scale.format<int>(), desc_scale);
        cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(tmp_zp.format<int>(), desc_scale);

        scales_block[0].format<half, 16, 16>().select<8, 2, 16, 1>(0) = tmp_scale.format<half, 8, 32>().select<8, 1, 16, 2>(0, 0);
        scales_block[0].format<half, 16, 16>().select<8, 2, 16, 1>(1) = tmp_scale.format<half, 8, 32>().select<8, 1, 16, 2>(0, 1);
        zps_block[0].format<half, 16, 16>().select<8, 2, 16, 1>(0) = tmp_zp.format<half, 8, 32>().select<8, 1, 16, 2>(0, 0);
        zps_block[0].format<half, 16, 16>().select<8, 2, 16, 1>(1) = tmp_zp.format<half, 8, 32>().select<8, 1, 16, 2>(0, 1);

        desc_scale.set_base(key_cache + scale_offset1);
        cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 0>(tmp_scale.format<int>(), desc_scale);
        cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(tmp_zp.format<int>(), desc_scale);
        scales_block[1].format<half, 16, 16>().select<8, 2, 16, 1>(0) = tmp_scale.format<half, 8, 32>().select<8, 1, 16, 2>(0, 0);
        scales_block[1].format<half, 16, 16>().select<8, 2, 16, 1>(1) = tmp_scale.format<half, 8, 32>().select<8, 1, 16, 2>(0, 1);
        zps_block[1].format<half, 16, 16>().select<8, 2, 16, 1>(0) = tmp_zp.format<half, 8, 32>().select<8, 1, 16, 2>(0, 0);
        zps_block[1].format<half, 16, 16>().select<8, 2, 16, 1>(1) = tmp_zp.format<half, 8, 32>().select<8, 1, 16, 2>(0, 1);
        #else
        matrix<half, KEY_LINES_PER_LOAD, STRIDE> tmp_scale, tmp_zp;  //KV_BLOCK_SIZE/2
        tmp_scale.format<uint>() = cm_load<uint, VectorSize::N8 /*(STRIDE / 2)*/>(key_cache, offsets_scale0); 
        scales_block[0].format<half, 16, 8>().select<8, 2, 8, 1>(0) = tmp_scale.format<half, 4, 32>().select<4, 1, 16, 2>(0, 0);
        scales_block[0].format<half, 16, 8>().select<8, 2, 8, 1>(1) = tmp_scale.format<half, 4, 32>().select<4, 1, 16, 2>(0, 1);
    
        vector<uint, KEY_LINES_PER_LOAD> offsets_scale0_ = offsets_scale0 + KEY_LINES_PER_LOAD*STRIDE * sizeof(half);
        tmp_scale.format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_scale0_);
        scales_block[1].format<half, 16, 8>().select<8, 2, 8, 1>(0) = tmp_scale.format<half, 4, 32>().select<4, 1, 16, 2>(0, 0);
        scales_block[1].format<half, 16, 8>().select<8, 2, 8, 1>(1) = tmp_scale.format<half, 4, 32>().select<4, 1, 16, 2>(0, 1);
        tmp_zp.format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_scale1);       
        zps_block[0].format<half, 16, 8>().select<8, 2, 8, 1>(0) = tmp_zp.format<half, 4, 32>().select<4, 1, 16, 2>(0, 0);
        zps_block[0].format<half, 16, 8>().select<8, 2, 8, 1>(1) = tmp_zp.format<half, 4, 32>().select<4, 1, 16, 2>(0, 1);

        vector<uint, KEY_LINES_PER_LOAD> offsets_scale1_ = offsets_scale1 + KEY_LINES_PER_LOAD*STRIDE * sizeof(half);
        tmp_zp.format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_scale1_);       
        zps_block[1].format<half, 16, 8>().select<8, 2, 8, 1>(0) = tmp_zp.format<half, 4, 32>().select<4, 1, 16, 2>(0, 0);
        zps_block[1].format<half, 16, 8>().select<8, 2, 8, 1>(1) = tmp_zp.format<half, 4, 32>().select<4, 1, 16, 2>(0, 1);

        #endif
    }
    #ifndef CM_HAS_LSC_UNTYPED_2D
    vector<uint, KEY_LINES_PER_LOAD> offsets_0;
    vector<uint, KEY_LINES_PER_LOAD> offsets_1;
    #endif
    if constexpr (IS64) {
        #ifdef CM_HAS_LSC_UNTYPED_2D
        cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0_up_s8.format<int>(), desc_b0);
        cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0_down_s8.format<int>(), desc_b1);
        #else
        offsets_0 = (gather_offsets_b) * pitch_b + off_b0;
        offsets_1 = (gather_offsets_b) * pitch_b + off_b1;
        b0_up_s8.format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_0);
        b0_down_s8.format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_1);
        #endif
    } else {
        #ifndef CM_HAS_LSC_UNTYPED_2D
        offsets_0 = (gather_offsets_b) * pitch_b + off_b0;
        offsets_1 = (gather_offsets_b) * pitch_b + off_b1;
        #endif
    }
#else
    #ifndef CM_HAS_LSC_UNTYPED_2D
    vector<uint, KEY_LINES_PER_LOAD> offsets_0;
    vector<uint, KEY_LINES_PER_LOAD> offsets_1;
    #endif
    if constexpr (IS64) {
        #ifdef CM_HAS_LSC_UNTYPED_2D
        cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0[0].format<int>(), desc_b0);
        cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0[1].format<int>(), desc_b1);
        #else
        offsets_0 = (gather_offsets_b) * pitch_b + off_b0;
        offsets_1 = (gather_offsets_b) * pitch_b + off_b1;
        b0[0].format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_0); 
        b0[1].format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_1);
        #endif
    } else {
        #ifndef CM_HAS_LSC_UNTYPED_2D
        offsets_0 = (gather_offsets_b) * pitch_b + off_b0;
        offsets_1 = (gather_offsets_b) * pitch_b + off_b1;
        #endif
    }
#endif
    if constexpr (IS64) {
        #ifdef CM_HAS_LSC_UNTYPED_2D
        desc_b0.set_block_x(desc_b0.get_block_x() + 8);
        desc_b1.set_block_x(desc_b1.get_block_x() + 8);
        #else
        offsets_0 += 8 * sizeof(uint); 
        offsets_1 += 8 * sizeof(uint);
        #endif
    }

#if USE_INT8
    auto dec = [&](vector<int, KEY_LINES_PER_LOAD*4> B0_i8, vector<int, KEY_LINES_PER_LOAD*4> B1_i8, matrix_ref<half, REG_N, BLOCK_REG_B> B0) {
#pragma unroll
        for (int n = 0; n < REG_N; n++) {
            auto b = B0[n].format<half, BLOCK_REG_K/2, BLOCK_REG_N*2>(); 
#pragma unroll
            for (int m = 0; m < BLOCK_REG_K/2; m++) {
                auto b_row = b[m];
                vector<ushort, BLOCK_REG_N> d0;
                if (n == 0)
                    d0 = B0_i8.format<ushort, 4, BLOCK_REG_N*2>()[m / 2].select<BLOCK_REG_N, 2>(m % 2);
                else
                    d0 = B1_i8.format<ushort, 4, BLOCK_REG_N*2>()[m / 2].select<BLOCK_REG_N, 2>(m % 2);
                b_row.format<ushort>() = d0.format<uchar>();
                b_row *= half{32768.0};
                b_row *= half{512.0};
                b_row = (b_row - zps[n]) * scales[n];
            }
        }
    };
#endif

    auto dot = [&](matrix<half, REG_M, BLOCK_REG_A> A, matrix<half, REG_N, BLOCK_REG_B> B) {
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
#pragma unroll
            for (uint reg_m = 0; reg_m < REG_M; reg_m++) {
                acc.row((ushort)(reg_m * REG_N + reg_n)) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(acc.row((ushort)(reg_m * REG_N + reg_n)),
                   B.row((ushort)reg_n).format<int>(), A.row((ushort)reg_m).format<int>());
            }
        }
    };

    for (uint s = 0; s < STRIDE; s++) {
#if USE_INT8
        auto tmp = scales_block[0].select<KEY_LINES_PER_LOAD, 1>(s * KEY_LINES_PER_LOAD);
        scales[0].select<KEY_LINES_PER_LOAD, 2>(0) = tmp;
        scales[0].select<KEY_LINES_PER_LOAD, 2>(1) = scales[0].select<KEY_LINES_PER_LOAD, 2>(0);
        tmp = scales_block[1].select<KEY_LINES_PER_LOAD, 1>(s * KEY_LINES_PER_LOAD);
        scales[1].select<KEY_LINES_PER_LOAD, 2>(0) = tmp;
        scales[1].select<KEY_LINES_PER_LOAD, 2>(1) = scales[1].select<KEY_LINES_PER_LOAD, 2>(0);
        tmp = zps_block[0].select<KEY_LINES_PER_LOAD, 1>(s * KEY_LINES_PER_LOAD);
        zps[0].select<KEY_LINES_PER_LOAD, 2>(0) = tmp;
        zps[0].select<KEY_LINES_PER_LOAD, 2>(1) = zps[0].select<KEY_LINES_PER_LOAD, 2>(0);
        tmp = zps_block[1].select<KEY_LINES_PER_LOAD, 1>(s * KEY_LINES_PER_LOAD);
        zps[1].select<KEY_LINES_PER_LOAD, 2>(0) = tmp;
        zps[1].select<KEY_LINES_PER_LOAD, 2>(1) = zps[1].select<KEY_LINES_PER_LOAD, 2>(0);
#endif
        #pragma unroll
        for (uint hs = 0; hs < HEAD_SIZE / BLOCK_WG_K; hs++) {
            if constexpr (IS64) {
                // --------------------------------------------- unroll 0 ?      -----------------------------
                // prefetch
                #ifdef CM_HAS_LSC_UNTYPED_2D
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
                desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
                if (hs == HEAD_SIZE / BLOCK_WG_K - 1)
                    desc_prefetch_a.set_block_x((STRIDE - 1 - s - 1) * b_adjacent_between_head);
                else
                    desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);
                #else
                //cm_prefetch<16, DataSize::U32, CacheHint::Cached, CacheHint::Cached>(key_cache, off_prefetch_b);
                cm_prefetch_2d<BLOCK_WG_N / SG_MN, 32>(key_cache,off_prefetch_b,pitch_b);
                cm_prefetch_2d<BLOCK_WG_M / SG_MN, 32>(query,off_prefetch_a,pitch_a);
                off_prefetch_b += 32 * sizeof(half);
                if (hs == HEAD_SIZE / BLOCK_WG_K - 1)
                    off_prefetch_a = base_off_prefetch_a + ((STRIDE - 1 - s - 1) * b_adjacent_between_head) * sizeof(half);
                else
                    off_prefetch_a += 32 * sizeof(half);
                #endif

                // load a: M[0:16*4]xK[0:16]
                #ifdef CM_HAS_LSC_UNTYPED_2D
                cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0,  0>(a0.select<4, 1, BLOCK_REG_A, 1>(0).format<half>(), desc_a);
                cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0, 32>(a0.select<4, 1, BLOCK_REG_A, 1>(4).format<half>(), desc_a);
                #else
                cm_load_2d(a0.format<half, BLOCK_SG_M, BLOCK_REG_K>(), query, off_a, pitch_a, (id_sg_mn==0 && s==0));

                uint query_header_xbase = hq * HEAD_SIZE;
                uint offset_a_x = ((STRIDE - 1) * b_adjacent_between_head  + query_header_xbase) * sizeof(half);

                matrix_ref<half, BLOCK_SG_M, BLOCK_REG_K> aa = a0.format<half, BLOCK_SG_M, BLOCK_REG_K>();

                #endif

                // load b: N[0:16*2]xK[16:32]
#if USE_INT8
                #ifdef CM_HAS_LSC_UNTYPED_2D
                cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b1_up_s8.format<int>(), desc_b0);
                cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b1_down_s8.format<int>(), desc_b1);
                #else
                b1_up_s8.format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_0);
                b1_down_s8.format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_1);
                #endif
                dec(b0_up_s8.format<int>().select<KEY_LINES_PER_LOAD*4, 1>(), b0_down_s8.format<int>().select<KEY_LINES_PER_LOAD*4, 1>(), b0);
                dot(a0, b0);
#else
                #ifdef CM_HAS_LSC_UNTYPED_2D
                cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b1[0].format<int>(), desc_b0);
                cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b1[1].format<int>(), desc_b1);
                #else

                b1[0].format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_0);
                b1[1].format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_1);
                #endif
                // if(id_sg_mn==0 && s==0) show(b0.format<half, 2, 16*16>());
                dot(a0, b0);
#endif
                #ifdef CM_HAS_LSC_UNTYPED_2D
                desc_a.set_block_x(desc_a.get_block_x() + BLOCK_REG_K);
                desc_b0.set_block_x(desc_b0.get_block_x() + 8);
                desc_b1.set_block_x(desc_b1.get_block_x() + 8);
                #else
                off_a += BLOCK_REG_K * sizeof(half);
                offsets_0 += 8 * sizeof(uint);   //16 fp16, 32 int8
                offsets_1 += 8 * sizeof(uint);
                #endif

                // --------------------------------------------- unroll 1 ?      -----------------------------

                // load a: M[0:16*4]xK[16:32]
                #ifdef CM_HAS_LSC_UNTYPED_2D
                cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0,  0>(a0.select<4, 1, BLOCK_REG_A, 1>(0).format<half>(), desc_a);
                cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0, 32>(a0.select<4, 1, BLOCK_REG_A, 1>(4).format<half>(), desc_a);
                #else
                cm_load_2d(a0.format<half, BLOCK_SG_M, BLOCK_REG_K>(), query, off_a, pitch_a);
                #endif

#if USE_INT8
                dec(b0_up_s8.format<int>().select<KEY_LINES_PER_LOAD*4, 1>(KEY_LINES_PER_LOAD*4), b0_down_s8.format<int>().select<KEY_LINES_PER_LOAD*4, 1>(KEY_LINES_PER_LOAD*4), b0);
                dot(a0, b0);
#else
                #ifdef CM_HAS_LSC_UNTYPED_2D
                cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0[0].format<int>(), desc_b0);
                cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0[1].format<int>(), desc_b1);
                desc_b0.set_block_x(desc_b0.get_block_x() + 8);
                desc_b1.set_block_x(desc_b1.get_block_x() + 8);
                #else
                b0[0].format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_0);
                b0[1].format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_1);
                offsets_0 += 8 * sizeof(uint);   //16 fp16, 32 int8
                offsets_1 += 8 * sizeof(uint);
                #endif
                // if(id_sg_mn==0 && s==0) show(b1.format<half, 2, 16*16>());
                dot(a0, b1);
#endif
                #ifdef CM_HAS_LSC_UNTYPED_2D
                desc_a.set_block_x(desc_a.get_block_x() + BLOCK_REG_K);
                #else
                off_a += BLOCK_REG_K * sizeof(half);
                #endif

                // --------------------------------------------- unroll 2 ?      -----------------------------

                // prefetch
                #ifdef CM_HAS_LSC_UNTYPED_2D
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
                desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
                desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);

                // load a: M[0:16*4]xK[32:48]
                cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0,  0>(a0.select<4, 1, BLOCK_REG_A, 1>(0).format<half>(), desc_a);
                cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0, 32>(a0.select<4, 1, BLOCK_REG_A, 1>(4).format<half>(), desc_a);
                #else
                cm_prefetch_2d<BLOCK_WG_N / SG_MN, 32>(key_cache,off_prefetch_b,pitch_b);
                cm_prefetch_2d<BLOCK_WG_M / SG_MN, 32>(query,off_prefetch_a,pitch_a);
                off_prefetch_b += 32 * sizeof(half);
                off_prefetch_a += 32 * sizeof(half);

                // load a: M[0:16*4]xK[32:48]
                cm_load_2d(a0.format<half, BLOCK_SG_M, BLOCK_REG_K>(), query, off_a, pitch_a);
                #endif

                // load b: N[0:16*2]xK[32:64]
#if USE_INT8
                #ifdef CM_HAS_LSC_UNTYPED_2D
                cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0_up_s8.format<int>(), desc_b0);
                cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0_down_s8.format<int>(), desc_b1);
                #else
                b0_up_s8.format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_0);
                b0_down_s8.format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_1);
                #endif
                dec(b1_up_s8.format<int>().select<KEY_LINES_PER_LOAD*4, 1>(), b1_down_s8.format<int>().select<KEY_LINES_PER_LOAD*4, 1>(), b0);
                dot(a0, b0);
#else
                #ifdef CM_HAS_LSC_UNTYPED_2D
                cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b1[0].format<int>(), desc_b0);
                cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b1[1].format<int>(), desc_b1);
                #else
                b1[0].format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_0);
                b1[1].format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_1);
                #endif
                dot(a0, b0);
#endif
                #ifdef CM_HAS_LSC_UNTYPED_2D
                desc_a.set_block_x(desc_a.get_block_x() + BLOCK_REG_K);
                desc_b0.set_block_x(desc_b0.get_block_x() + 8);
                desc_b1.set_block_x(desc_b1.get_block_x() + 8);
                #else
                off_a += BLOCK_REG_K * sizeof(half);
                offsets_0 += 8 * sizeof(uint);   //16 fp16, 32 int8
                offsets_1 += 8 * sizeof(uint);
                #endif

                // --------------------------------------------- unroll 3 ?      -----------------------------

                // load a: M[0:16*4]xK[48:64]
                #ifdef CM_HAS_LSC_UNTYPED_2D
                cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0,  0>(a0.select<4, 1, BLOCK_REG_A, 1>(0).format<half>(), desc_a);
                cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0, 32>(a0.select<4, 1, BLOCK_REG_A, 1>(4).format<half>(), desc_a);
                if (hs == HEAD_SIZE / BLOCK_WG_K - 1) {
                    desc_a.set_block_x((STRIDE - 1 - s - 1) * b_adjacent_between_head);
                } else {
                    desc_a.set_block_x(desc_a.get_block_x() + BLOCK_REG_K);
                }
                #else
                cm_load_2d(a0.format<half, BLOCK_SG_M, BLOCK_REG_K>(), query, off_a, pitch_a);
                if (hs == HEAD_SIZE / BLOCK_WG_K - 1) {
                    off_a = base_off_a + ((STRIDE - 1 - s - 1) * b_adjacent_between_head) * sizeof(half);
                } else {
                    off_a += BLOCK_REG_K * sizeof(half);
                }
                #endif
#if USE_INT8
                dec(b1_up_s8.format<int>().select<KEY_LINES_PER_LOAD*4, 1>(KEY_LINES_PER_LOAD*4), b1_down_s8.format<int>().select<KEY_LINES_PER_LOAD*4, 1>(KEY_LINES_PER_LOAD*4), b0);
                dot(a0, b0);
#else
                #ifdef CM_HAS_LSC_UNTYPED_2D
                // load b: N[0:16*2]xK[16:32]
                cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0[0].format<int>(), desc_b0);
                cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0[1].format<int>(), desc_b1);
                dot(a0, b0);
                desc_b0.set_block_x(desc_b0.get_block_x() + 8);
                desc_b1.set_block_x(desc_b1.get_block_x() + 8);
                #else
                b0[0].format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_0);
                b0[1].format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_1);
                offsets_0 += 8 * sizeof(uint); 
                offsets_1 += 8 * sizeof(uint);
                #endif
                dot(a0, b1);
#endif
            } else {
                #ifdef CM_HAS_LSC_UNTYPED_2D
                // 2 phases with single buffer b0
                // prefetch
                desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 16);
                desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 16);
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);

                // load a: M[0:16*4]xK[0:16]
                cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0,  0>(a0.select<4, 1, BLOCK_REG_A, 1>(0).format<half>(), desc_a);
                cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0, 32>(a0.select<4, 1, BLOCK_REG_A, 1>(4).format<half>(), desc_a);
#if USE_INT8
                // load b: N[0:16*2]xK[0:32]
                cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0_up_s8.format<int>(), desc_b0);
                cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0_down_s8.format<int>(), desc_b1);
                dec(b0_up_s8.format<int>().select<64, 1>(), b0_down_s8.format<int>().select<64, 1>(), b0);
                dot(a0, b0);
#else
                // load b: N[0:16*2]xK[0:16]
                cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0[0].format<int>(), desc_b0);
                cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0[1].format<int>(), desc_b1);
                dot(a0, b0);
#endif

                desc_a.set_block_x(desc_a.get_block_x() + 16);
                desc_b0.set_block_x(desc_b0.get_block_x() + 8);
                desc_b1.set_block_x(desc_b1.get_block_x() + 8);

                // prefetch
                desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 16);
                if (hs == HEAD_SIZE / BLOCK_WG_K - 1)
                    desc_prefetch_a.set_block_x((STRIDE - 1 - s - 1) * b_adjacent_between_head);
                else
                    desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 16);
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
                cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);

                // load a: M[0:16*4]xK[16:32]
                cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0,  0>(a0.select<4, 1, BLOCK_REG_A, 1>(0).format<half>(), desc_a);
                cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0, 32>(a0.select<4, 1, BLOCK_REG_A, 1>(4).format<half>(), desc_a);
                if (hs == HEAD_SIZE / BLOCK_WG_K - 1) {
                    desc_a.set_block_x((STRIDE - 1 - s - 1) * b_adjacent_between_head);
                } else {
                    desc_a.set_block_x(desc_a.get_block_x() + 16);
                }
#if USE_INT8
                dec(b0_up_s8.format<int>().select<64, 1>(64), b0_down_s8.format<int>().select<64, 1>(64), b0);
                dot(a0, b0);
#else
                // load b: N[0:16*2]xK[16:32]
                cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0[0].format<int>(), desc_b0);
                cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0[1].format<int>(), desc_b1);
                dot(a0, b0);
                desc_b0.set_block_x(desc_b0.get_block_x() + 8);
                desc_b1.set_block_x(desc_b1.get_block_x() + 8);
#endif
                #else
                // 2 phases with single buffer b0 (non-lsc)

                // prefetch
                cm_prefetch_2d<BLOCK_WG_N / SG_MN, PHASE_SIZE>(key_cache, off_prefetch_b, pitch_b);
                cm_prefetch_2d<BLOCK_WG_M / SG_MN, PHASE_SIZE>(query, off_prefetch_a, pitch_a);
                off_prefetch_b += PHASE_SIZE * sizeof(half);
                off_prefetch_a += PHASE_SIZE * sizeof(half);

                // load a: M[0:16*4]xK[0:16]
                cm_load_2d(a0.format<half, BLOCK_SG_M, BLOCK_REG_K>(), query, off_a, pitch_a);

#if USE_INT8
                // load b once: N[0:16*2]xK[0:32], then split into 2 phases
                b0_up_s8.format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_0);
                b0_down_s8.format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_1);
                dec(b0_up_s8.format<int>().select<KEY_LINES_PER_LOAD * 4, 1>(), b0_down_s8.format<int>().select<KEY_LINES_PER_LOAD * 4, 1>(), b0);
                dot(a0, b0);
#else
                // load b: N[0:16*2]xK[0:16]
                b0[0].format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_0);
                b0[1].format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_1);
                dot(a0, b0);
#endif

                off_a += PHASE_SIZE * sizeof(half);
                offsets_0 += 8 * sizeof(uint);
                offsets_1 += 8 * sizeof(uint);

                // prefetch
                cm_prefetch_2d<BLOCK_WG_N / SG_MN, PHASE_SIZE>(key_cache, off_prefetch_b, pitch_b);
                cm_prefetch_2d<BLOCK_WG_M / SG_MN, PHASE_SIZE>(query, off_prefetch_a, pitch_a);
                off_prefetch_b += PHASE_SIZE * sizeof(half);
                if (hs == HEAD_SIZE / BLOCK_WG_K - 1)
                    off_prefetch_a = base_off_prefetch_a + ((STRIDE - 1 - s - 1) * b_adjacent_between_head) * sizeof(half);
                else
                    off_prefetch_a += PHASE_SIZE * sizeof(half);

                // load a: M[0:16*4]xK[16:32]
                cm_load_2d(a0.format<half, BLOCK_SG_M, BLOCK_REG_K>(), query, off_a, pitch_a);
                if (hs == HEAD_SIZE / BLOCK_WG_K - 1) {
                    off_a = base_off_a + ((STRIDE - 1 - s - 1) * b_adjacent_between_head) * sizeof(half);
                } else {
                    off_a += PHASE_SIZE * sizeof(half);
                }

#if USE_INT8
                dec(b0_up_s8.format<int>().select<KEY_LINES_PER_LOAD * 4, 1>(KEY_LINES_PER_LOAD * 4), b0_down_s8.format<int>().select<KEY_LINES_PER_LOAD * 4, 1>(KEY_LINES_PER_LOAD * 4), b0);
                dot(a0, b0);
#else
                // load b: N[0:16*2]xK[16:32]
                b0[0].format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_0);
                b0[1].format<uint>() = cm_load<uint, VectorSize::N8>(key_cache, offsets_1);
                dot(a0, b0);
#endif
                offsets_0 += 8 * sizeof(uint);
                offsets_1 += 8 * sizeof(uint);
                #endif
            }
        }
    }

    cm_barrier();

    matrix<SOFTMAX_TYPE, REG_M * BLOCK_REG_M, REG_N * BLOCK_REG_N> acc_half;
    union {
        float f;
        int y;
    } i2f;
    i2f.y = INV_S;
    const float inv_s = i2f.f;
#pragma unroll
    for (uint reg_m = 0; reg_m < REG_M; reg_m++) {
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
            acc_half.select<BLOCK_REG_M, 1, BLOCK_REG_N, 1>(reg_m * BLOCK_REG_M, reg_n * BLOCK_REG_N) =
                acc.row(reg_m * REG_N + reg_n) * inv_s; 
        }
    }

    //if(id_sg_m==0 && id_sg_n==0) show(acc.format<float, REG_M * REG_N, BLOCK_DPAS_C>());
    // if(id_sg_m==0 && id_sg_n==0) show(acc_half.format<float, REG_M * BLOCK_REG_M, REG_N * BLOCK_REG_N>());

    // if M(aka query) has tails, the following will not change the accuracy:
    //    gemm will compute results for the padding M(all should be zeros), the kq_max/kq_max_wg/kq_exp_partial_sum are along the query dimension and
    //    the results can be dropped in the future stage. To simplify the logic, the size of kq_max/kq_max_wg/kq_exp_partial_sum must be enough to hold
    //    all tails + padding results.
    int n_start = (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N);
    n_start = MYMIN(n_start, N);
    int n_end = MYMIN(n_start + BLOCK_SG_N, N);
    int valid_n = n_end - n_start;
    matrix<SOFTMAX_TYPE, BLOCK_SG_M, SUM_N> sum_t;
    vector<int, BLOCK_SG_M> seq_m;
    cmtl::cm_vector_assign(seq_m.select_all(), 0, 1);
    vector_ref<int, BLOCK_SG_N> seq = seq_m.select<BLOCK_SG_N, 1>();
    vector<uint, BLOCK_SG_N> n_pos = (uint)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) + seq;
#if IS_CAUSAL == 1
    bool skip_mask = false;
    int m_start = (int)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M + q_start_strided);
    // in streaming scenario, the past kvcache length may be arbitrary so valid causal mask of a workgroup may start at arbitrary position
    if (n_end <= m_start) {
        // all are inside causal mask == 1
        skip_mask = true;
    } else {
    #pragma unroll
        for (uint reg_m = 0; reg_m < REG_M * BLOCK_REG_M; reg_m++) {
            SIMD_IF_BEGIN (n_pos > m_start + reg_m) {
                acc_half.row(reg_m) = SOFTMAX_TYPE{-60000};
            } SIMD_IF_END;
        }
    }
#else
    bool skip_mask = true;
#endif
    vector<SOFTMAX_TYPE, BLOCK_SG_M> max_m;
    if (valid_n != BLOCK_SG_N) {
#pragma unroll
        for (uint reg_m = 0; reg_m < REG_M * BLOCK_REG_M; reg_m++) {
            acc_half.row(reg_m).merge(SOFTMAX_TYPE{-60000}, n_pos >= N);
        }
    }

        max_m.select<BLOCK_SG_M, 1>() = reduce2d<1, 0, 1>(acc_half.select<BLOCK_SG_M, 1, BLOCK_SG_N, 1>()).format<SOFTMAX_TYPE>();

    {
        uint slm_offset = (id_sg_n * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) * (uint)sizeof(SOFTMAX_TYPE);
        // current max -> slm
        cm_slm_block_write(slm, slm_offset, max_m.format<int>());
        cm_slm_fence(CM_LOCAL_BARRIER);
        cm_barrier();
        // max inside wg
        cm_slm_block_read(slm, id_sg_m * BLOCK_SG_M * (uint)sizeof(SOFTMAX_TYPE), max_m.format<int>());
        vector<SOFTMAX_TYPE, BLOCK_SG_M> tmp;
#pragma unroll
        for (uint i = 1; i < SG_N; i++) {
            slm_offset = (i * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) * (uint)sizeof(SOFTMAX_TYPE);
            cm_slm_block_read(slm, slm_offset, tmp.format<int>());
            max_m = cm_max<SOFTMAX_TYPE>(max_m, tmp);
        }

        // if(id_sg_m==0 && id_sg_n==0) 
        // show(max_m.format<SOFTMAX_TYPE, 1, BLOCK_SG_M>());

        // max across wg
        // kq_max: [b, hq, M_aligned]
        //auto max_offsets = (id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M + seq_m) * (uint)sizeof(half);
        //cm_ptr_atomic<AtomicOp::FMAX, half>((half*)kq_max, max_offsets, max_m);

        // current max -> mem
        // kq_max_wg: [b, hq, N/BLOCK_WG_N, M_aligned]
        uint offset = (id_wg_n * M_aligned + id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) * sizeof(SOFTMAX_TYPE);
        cm_ptr_store<int>((int*)kq_max_wg, offset, max_m.format<int>());
    }
    
    {
        // kq_exp_partial_sum: [b, hq, M_aligned, N/(BLOCK_SIZE/STRIDE)]
        if (valid_n == BLOCK_SG_N && skip_mask) {
#pragma unroll
            for (uint reg_m = 0; reg_m < REG_M * BLOCK_REG_M; reg_m++) {
                acc_half.row(reg_m) = cm_exp((acc_half.row(reg_m) - max_m[reg_m]) * log2e);
            }
        } else {
#pragma unroll
            for (uint reg_m = 0; reg_m < REG_M * BLOCK_REG_M; reg_m++) {
                acc_half.row(reg_m) = cm_exp((acc_half.row(reg_m) - max_m[reg_m]) * log2e);
                // causal mask in the following case:
                // block0(EU0)   block1(EU1)
                // 1 1 1 1       0 0 0 0
                // 1 1 1 1       1 0 0 0
                // 1 1 1 1       1 1 0 0
                // 1 1 1 1       1 1 1 0
                // the acc value of first row of block1 should be -inf --> max(first column) == -inf --> exp(first column - max) == 1, this is incorrect
                // so need to use simd_if to detect per element state
#if IS_CAUSAL
                SIMD_IF_BEGIN ((n_pos > m_start + reg_m) | (n_pos >= N)) {
#else
                SIMD_IF_BEGIN (n_pos >= N) {
#endif
                    acc_half.row(reg_m) = 0;
                } SIMD_IF_END;
            }
        }
        #ifdef CM_HAS_LSC_UNTYPED_2D
        sum_t.select<32, 1, SUM_N, 1>(0).format<SOFTMAX_TYPE>() = reduce2d<SUM_N, 1, SUM_N>(acc_half.select<32, 1, BLOCK_SG_N, 1>(0)).format<SOFTMAX_TYPE>();
        sum_t.select<32, 1, SUM_N, 1>(32).format<SOFTMAX_TYPE>() = reduce2d<SUM_N, 1, SUM_N>(acc_half.select<32, 1, BLOCK_SG_N, 1>(32)).format<SOFTMAX_TYPE>();
        #else
        sum_t.select<BLOCK_SG_M, 1, SUM_N, 1>(0).format<SOFTMAX_TYPE>() = reduce2d<SUM_N, 1, SUM_N>(acc_half.select<BLOCK_SG_M, 1, BLOCK_SG_N, 1>(0)).format<SOFTMAX_TYPE>();
        #endif
    }
    // store
    #ifdef CM_HAS_LSC_UNTYPED_2D
    lsc::block_2d_desc<SOFTMAX_TYPE, 1, 8, k_block_in_sg> desc_c{ kq_exp_partial_sum, M - 1 /*height*/, (uint)(K_block_pad * sizeof(SOFTMAX_TYPE) - 1) /*width*/, (uint)(K_block_pad * sizeof(SOFTMAX_TYPE) - 1),
        (int)((id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) / block_size_div_stride) /*x*/, (int)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) /*y*/ };
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 0>(desc_c, sum_t.select<8, 1, k_block_in_sg, 1>( 0).format<SOFTMAX_TYPE>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 1>(desc_c, sum_t.select<8, 1, k_block_in_sg, 1>( 8).format<SOFTMAX_TYPE>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 2>(desc_c, sum_t.select<8, 1, k_block_in_sg, 1>(16).format<SOFTMAX_TYPE>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 3>(desc_c, sum_t.select<8, 1, k_block_in_sg, 1>(24).format<SOFTMAX_TYPE>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 4>(desc_c, sum_t.select<8, 1, k_block_in_sg, 1>(32).format<SOFTMAX_TYPE>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 5>(desc_c, sum_t.select<8, 1, k_block_in_sg, 1>(40).format<SOFTMAX_TYPE>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 6>(desc_c, sum_t.select<8, 1, k_block_in_sg, 1>(48).format<SOFTMAX_TYPE>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 7>(desc_c, sum_t.select<8, 1, k_block_in_sg, 1>(56).format<SOFTMAX_TYPE>());
    #else
    const uint pitch_c = K_block_pad * sizeof(SOFTMAX_TYPE);
    uint off_c = offset_partial_sum;
    off_c += pitch_c * (id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M);
    off_c += ((id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) / block_size_div_stride) * sizeof(SOFTMAX_TYPE);

    #pragma unroll
    for (uint i = 0; i < BLOCK_SG_M/8; i++) {
        cm_store_2d(sum_t.select<8, 1, SUM_N, 1>( i*8).format<uint, 8, SUM_N>(), kq_exp_partial_sum, off_c + (8 * i) * pitch_c, pitch_c);
    }

    #endif
}
#endif
