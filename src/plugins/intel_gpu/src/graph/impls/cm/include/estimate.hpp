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

template <typename T1, typename T2>
CM_INLINE void Transpose_8x8(matrix_ref<T1, 8, 8> in, matrix_ref<T2, 8, 8> out) {
    matrix<T2, 8, 8> temp;
    temp.row(0) = in.template select<2, 1, 4, 2>(0, 0);
    temp.row(1) = in.template select<2, 1, 4, 2>(2, 0);
    temp.row(2) = in.template select<2, 1, 4, 2>(4, 0);
    temp.row(3) = in.template select<2, 1, 4, 2>(6, 0);
    temp.row(4) = in.template select<2, 1, 4, 2>(0, 1);
    temp.row(5) = in.template select<2, 1, 4, 2>(2, 1);
    temp.row(6) = in.template select<2, 1, 4, 2>(4, 1);
    temp.row(7) = in.template select<2, 1, 4, 2>(6, 1);

    out.row(0) = temp.template select<4, 1, 2, 4>(0, 0);
    out.row(2) = temp.template select<4, 1, 2, 4>(0, 1);
    out.row(4) = temp.template select<4, 1, 2, 4>(0, 2);
    out.row(6) = temp.template select<4, 1, 2, 4>(0, 3);
    out.row(1) = temp.template select<4, 1, 2, 4>(4, 0);
    out.row(3) = temp.template select<4, 1, 2, 4>(4, 1);
    out.row(5) = temp.template select<4, 1, 2, 4>(4, 2);
    out.row(7) = temp.template select<4, 1, 2, 4>(4, 3);
}

template <typename T1, typename T2>
CM_INLINE void Transpose_8x32(matrix_ref<T1, 8, 32> in, matrix_ref<T2, 32, 8> out) {
    Transpose_8x8(in.template select<8, 1, 8, 1>(0,  0), out.template select<8, 1, 8, 1>( 0, 0));
    Transpose_8x8(in.template select<8, 1, 8, 1>(0,  8), out.template select<8, 1, 8, 1>( 8, 0));
    Transpose_8x8(in.template select<8, 1, 8, 1>(0, 16), out.template select<8, 1, 8, 1>(16, 0));
    Transpose_8x8(in.template select<8, 1, 8, 1>(0, 24), out.template select<8, 1, 8, 1>(24, 0));
}

template <typename T1, typename T2>
CM_INLINE void Transpose_4x32(matrix_ref<T1, 4, 32> in, matrix_ref<T2, 32, 4> out) {
    matrix<T2, 4, 32> temp;
    temp.row(0) = in.template select<4, 1, 8, 4>(0, 0);
    temp.row(1) = in.template select<4, 1, 8, 4>(0, 1);
    temp.row(2) = in.template select<4, 1, 8, 4>(0, 2);
    temp.row(3) = in.template select<4, 1, 8, 4>(0, 3);

    out.row( 0) = temp.template select<1, 1, 4, 8>(0, 0);
    out.row( 1) = temp.template select<1, 1, 4, 8>(1, 0);
    out.row( 2) = temp.template select<1, 1, 4, 8>(2, 0);
    out.row( 3) = temp.template select<1, 1, 4, 8>(3, 0);
    out.row( 4) = temp.template select<1, 1, 4, 8>(0, 1);
    out.row( 5) = temp.template select<1, 1, 4, 8>(1, 1);
    out.row( 6) = temp.template select<1, 1, 4, 8>(2, 1);
    out.row( 7) = temp.template select<1, 1, 4, 8>(3, 1);
    out.row( 8) = temp.template select<1, 1, 4, 8>(0, 2);
    out.row( 9) = temp.template select<1, 1, 4, 8>(1, 2);
    out.row(10) = temp.template select<1, 1, 4, 8>(2, 2);
    out.row(11) = temp.template select<1, 1, 4, 8>(3, 2);
    out.row(12) = temp.template select<1, 1, 4, 8>(0, 3);
    out.row(13) = temp.template select<1, 1, 4, 8>(1, 3);
    out.row(14) = temp.template select<1, 1, 4, 8>(2, 3);
    out.row(15) = temp.template select<1, 1, 4, 8>(3, 3);
    out.row(16) = temp.template select<1, 1, 4, 8>(0, 4);
    out.row(17) = temp.template select<1, 1, 4, 8>(1, 4);
    out.row(18) = temp.template select<1, 1, 4, 8>(2, 4);
    out.row(19) = temp.template select<1, 1, 4, 8>(3, 4);
    out.row(20) = temp.template select<1, 1, 4, 8>(0, 5);
    out.row(21) = temp.template select<1, 1, 4, 8>(1, 5);
    out.row(22) = temp.template select<1, 1, 4, 8>(2, 5);
    out.row(23) = temp.template select<1, 1, 4, 8>(3, 5);
    out.row(24) = temp.template select<1, 1, 4, 8>(0, 6);
    out.row(25) = temp.template select<1, 1, 4, 8>(1, 6);
    out.row(26) = temp.template select<1, 1, 4, 8>(2, 6);
    out.row(27) = temp.template select<1, 1, 4, 8>(3, 6);
    out.row(28) = temp.template select<1, 1, 4, 8>(0, 7);
    out.row(29) = temp.template select<1, 1, 4, 8>(1, 7);
    out.row(30) = temp.template select<1, 1, 4, 8>(2, 7);
    out.row(31) = temp.template select<1, 1, 4, 8>(3, 7);
}

template <typename T1, typename T2>
CM_INLINE void Transpose_32x32(matrix_ref<T1, 32, 32> in, matrix_ref<T2, 32, 32> out) {
    matrix<T2, 32, 32> temp;
    temp.row( 0) = in.template select<8, 1, 4, 8>( 0, 0);
    temp.row( 1) = in.template select<8, 1, 4, 8>( 8, 0);
    temp.row( 2) = in.template select<8, 1, 4, 8>(16, 0);
    temp.row( 3) = in.template select<8, 1, 4, 8>(24, 0);
    temp.row( 4) = in.template select<8, 1, 4, 8>( 0, 1);
    temp.row( 5) = in.template select<8, 1, 4, 8>( 8, 1);
    temp.row( 6) = in.template select<8, 1, 4, 8>(16, 1);
    temp.row( 7) = in.template select<8, 1, 4, 8>(24, 1);
    temp.row( 8) = in.template select<8, 1, 4, 8>( 0, 2);
    temp.row( 9) = in.template select<8, 1, 4, 8>( 8, 2);
    temp.row(10) = in.template select<8, 1, 4, 8>(16, 2);
    temp.row(11) = in.template select<8, 1, 4, 8>(24, 2);
    temp.row(12) = in.template select<8, 1, 4, 8>( 0, 3);
    temp.row(13) = in.template select<8, 1, 4, 8>( 8, 3);
    temp.row(14) = in.template select<8, 1, 4, 8>(16, 3);
    temp.row(15) = in.template select<8, 1, 4, 8>(24, 3);
    temp.row(16) = in.template select<8, 1, 4, 8>( 0, 4);
    temp.row(17) = in.template select<8, 1, 4, 8>( 8, 4);
    temp.row(18) = in.template select<8, 1, 4, 8>(16, 4);
    temp.row(19) = in.template select<8, 1, 4, 8>(24, 4);
    temp.row(20) = in.template select<8, 1, 4, 8>( 0, 5);
    temp.row(21) = in.template select<8, 1, 4, 8>( 8, 5);
    temp.row(22) = in.template select<8, 1, 4, 8>(16, 5);
    temp.row(23) = in.template select<8, 1, 4, 8>(24, 5);
    temp.row(24) = in.template select<8, 1, 4, 8>( 0, 6);
    temp.row(25) = in.template select<8, 1, 4, 8>( 8, 6);
    temp.row(26) = in.template select<8, 1, 4, 8>(16, 6);
    temp.row(27) = in.template select<8, 1, 4, 8>(24, 6);
    temp.row(28) = in.template select<8, 1, 4, 8>( 0, 7);
    temp.row(29) = in.template select<8, 1, 4, 8>( 8, 7);
    temp.row(30) = in.template select<8, 1, 4, 8>(16, 7);
    temp.row(31) = in.template select<8, 1, 4, 8>(24, 7);

    out.row( 0) = temp.template select<4, 1, 8, 4>( 0, 0);
    out.row( 1) = temp.template select<4, 1, 8, 4>( 4, 0);
    out.row( 2) = temp.template select<4, 1, 8, 4>( 8, 0);
    out.row( 3) = temp.template select<4, 1, 8, 4>(12, 0);
    out.row( 4) = temp.template select<4, 1, 8, 4>(16, 0);
    out.row( 5) = temp.template select<4, 1, 8, 4>(20, 0);
    out.row( 6) = temp.template select<4, 1, 8, 4>(24, 0);
    out.row( 7) = temp.template select<4, 1, 8, 4>(28, 0);
    out.row( 8) = temp.template select<4, 1, 8, 4>( 0, 1);
    out.row( 9) = temp.template select<4, 1, 8, 4>( 4, 1);
    out.row(10) = temp.template select<4, 1, 8, 4>( 8, 1);
    out.row(11) = temp.template select<4, 1, 8, 4>(12, 1);
    out.row(12) = temp.template select<4, 1, 8, 4>(16, 1);
    out.row(13) = temp.template select<4, 1, 8, 4>(20, 1);
    out.row(14) = temp.template select<4, 1, 8, 4>(24, 1);
    out.row(15) = temp.template select<4, 1, 8, 4>(28, 1);
    out.row(16) = temp.template select<4, 1, 8, 4>( 0, 2);
    out.row(17) = temp.template select<4, 1, 8, 4>( 4, 2);
    out.row(18) = temp.template select<4, 1, 8, 4>( 8, 2);
    out.row(19) = temp.template select<4, 1, 8, 4>(12, 2);
    out.row(20) = temp.template select<4, 1, 8, 4>(16, 2);
    out.row(21) = temp.template select<4, 1, 8, 4>(20, 2);
    out.row(22) = temp.template select<4, 1, 8, 4>(24, 2);
    out.row(23) = temp.template select<4, 1, 8, 4>(28, 2);
    out.row(24) = temp.template select<4, 1, 8, 4>( 0, 3);
    out.row(25) = temp.template select<4, 1, 8, 4>( 4, 3);
    out.row(26) = temp.template select<4, 1, 8, 4>( 8, 3);
    out.row(27) = temp.template select<4, 1, 8, 4>(12, 3);
    out.row(28) = temp.template select<4, 1, 8, 4>(16, 3);
    out.row(29) = temp.template select<4, 1, 8, 4>(20, 3);
    out.row(30) = temp.template select<4, 1, 8, 4>(24, 3);
    out.row(31) = temp.template select<4, 1, 8, 4>(28, 3);
}

// group_count: M = group_count * group_size, group_size is the element count of current reduction
// op: 0-max, 1-sum
// M: before reduce element count, N: row count, stop: element count M must be larger than
template <int group_count, int op, int stop, typename T, int N, int M>
CM_INLINE constexpr auto reduce2d(matrix_ref<T, N, M> src) {
    constexpr int group_size = M / group_count;
    if constexpr (N > stop) {
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


#if USE_KQ == 1 && (BLOCK_SG_M == 64 && BLOCK_SG_N == 32)
// register tile: [8, 2] aka[(8*8,16), (16, 16*2)]
// src_a is key, src_b is query
CM_INLINE void gemm_kq_64x32_xe2(uint id_wg_m, uint id_wg_n, uint hq, uint slm, svmptr_t key_cache, svmptr_t query, svmptr_t block_indices ATTR, svmptr_t block_indices_begins ATTR, svmptr_t kq_max ATTR, svmptr_t kq_max_wg ATTR, svmptr_t kq_exp_partial_sum ATTR,
uint M, uint N, uint K, uint query_stride, uint q_start_strided) {
    constexpr int SG_SIZE = 16;
    constexpr int BLOCK_WG_K = 64;	// same in sg
#ifndef BLOCK_SG_M
    #define BLOCK_SG_M  64
    #define BLOCK_SG_N  32
    #define SG_M  4
    #define SG_N  4
    #define HEAD_SIZE  128
    #define KV_BLOCK_SIZE  256
    #define STRIDE  16
#endif
    // xehpg DPAS spec: dst: [8, 8], repeat: 1~8, depth: 8
    static constexpr int REPEAT = 8;
    static constexpr int DEPTH = 8;
    static constexpr int BLOCK_REG_M = REPEAT;
    static constexpr int BLOCK_REG_N = SG_SIZE;
    static constexpr int BLOCK_DPAS_C = BLOCK_REG_M * BLOCK_REG_N;
    static constexpr int VNNI = sizeof(half);
    static constexpr int BLOCK_REG_K = DEPTH * sizeof(int) / VNNI;
    static constexpr int BLOCK_REG_A = BLOCK_REG_M * BLOCK_REG_K;
    static constexpr int BLOCK_REG_B = BLOCK_REG_N * BLOCK_REG_K;
    static constexpr int BLOCK_WG_M = SG_M * BLOCK_SG_M;
    static constexpr int BLOCK_WG_N = SG_N * BLOCK_SG_N;
    // register blocking
    static constexpr int REG_M = BLOCK_SG_M / BLOCK_REG_M;
    static constexpr int REG_N = BLOCK_SG_N / BLOCK_REG_N;
    static constexpr int REG_K = BLOCK_WG_K / BLOCK_REG_K;
    static constexpr int REG_MN = REG_M * REG_N;
    static constexpr int KEY_LINES_PER_LOAD = KV_BLOCK_SIZE / STRIDE;

    matrix<float, REG_M * REG_N, BLOCK_DPAS_C> acc = 0;                              // --> 64*2 regs
    uint id_sg_n = cm_local_id(0);
    uint id_sg_m = cm_local_id(1);
    uint id_sg_mn = id_sg_m * SG_N + id_sg_n;

    static_assert(REG_N == 2, "block_2d_desc for b is manually unrolled by 2");
    static_assert(HEAD_SIZE % BLOCK_WG_K == 0, "K dimension must be multiple of BLOCK_WG_K");
    static_assert(KV_BLOCK_SIZE == 256, "block size of key(key_cache) should be 256");
    uint M_block = (M + BLOCK_WG_M - 1) / BLOCK_WG_M;
    uint N_aligned = (N + BLOCK_WG_N - 1) / BLOCK_WG_N * BLOCK_WG_N;
    uint M_block_aligned = M_block * (BLOCK_WG_M / (BLOCK_SIZE / STRIDE));
    const uint block_size_div_stride = BLOCK_SIZE / STRIDE;
    constexpr half log2e = 1.4426950408889634f;
    static_assert(BLOCK_SG_M / block_size_div_stride == 8, "BLOCK_SG_M / block_size_div_stride should be 8");
    static_assert(BLOCK_SG_N == 32, "BLOCK_SG_N should be 32");

#if IS_CAUSAL == 1
    if ((int)(id_wg_m * BLOCK_WG_M) >= ((int)id_wg_n + 1) * BLOCK_WG_N + q_start_strided) {
        // fill -inf -> max in group, 0 -> exp_sum to make compensation work
        {
            // current max -> mem
            vector<half, BLOCK_SG_N> max_n = -60000;
            // kq_max_wg: [b, hq, M/BLOCK_WG_M, N_aligned]
            uint offset = (id_wg_m * N_aligned + id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) * sizeof(half);
            cm_ptr_store<int>((int*)kq_max_wg, offset, max_n.format<int>());
        }
        {
            // store
            matrix<half, 8, 8> sum_t = 0;
            lsc::block_2d_desc<half, 1, 8, 8> desc_c{ kq_exp_partial_sum, N - 1, (uint)(M_block_aligned * sizeof(half) - 1), (uint)(M_block_aligned * sizeof(half) - 1),
                (int)((id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) / block_size_div_stride), (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) };
            cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 0>(desc_c, sum_t.format<half>());
            cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 1>(desc_c, sum_t.format<half>());
            cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 2>(desc_c, sum_t.format<half>());
            cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 3>(desc_c, sum_t.format<half>());
        }

        return;
    }
#endif
    // assume block index coming from 0 in block_indices_begins
    int block_index_begin = ((int*)block_indices_begins)[0];
    int* block_indices_p = (int*)block_indices + block_index_begin;
    int b_adjacent_between_head = query_stride / STRIDE;
    // N[0:16*2]xK[0:16]
    lsc::block_2d_desc<int, 1, BLOCK_REG_N, BLOCK_REG_K / 2> desc_b0{ query, N - 1, (uint)((query_stride - hq * HEAD_SIZE) * sizeof(half) - 1), (uint)(query_stride * sizeof(half) - 1),
        (STRIDE - 1) * b_adjacent_between_head / 2, (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) };
    // prefetch B
    static constexpr int SG_MN = SG_M * SG_N;
    lsc::block_2d_desc<half, 1, BLOCK_WG_N / SG_MN, 32> desc_prefetch_b{query, N - 1, (uint)((query_stride - hq * HEAD_SIZE) * sizeof(half) - 1), (uint)(query_stride * sizeof(half) - 1),
        (STRIDE - 1) * b_adjacent_between_head, (int)(id_wg_n * BLOCK_WG_N + id_sg_mn * (BLOCK_WG_N / SG_MN)) };
    // N[0:16*2]xK[0:16]                                                                  --> 8+8 regs
    matrix<half, REG_N, BLOCK_REG_B> b0, b1;

    // M[0:16]xK[0:32]
    uint block_idx = (uint)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) * STRIDE / KV_BLOCK_SIZE;
    uint offset = block_indices_p[block_idx] * (HK * KV_BLOCK_SIZE * HEAD_SIZE * (uint)sizeof(half));
    lsc::block_2d_desc<half, 2, KEY_LINES_PER_LOAD, BLOCK_REG_K> desc_a0{ key_cache + offset, KEY_LINES_PER_LOAD - 1, (uint)(K * sizeof(half) - 1), (uint)(K * sizeof(half) - 1),
        0, 0 };
    // M[16:32]xK[0:32]
    offset = block_indices_p[block_idx + 1] * (HK * KV_BLOCK_SIZE * HEAD_SIZE * (uint)sizeof(half));
    lsc::block_2d_desc<half, 2, KEY_LINES_PER_LOAD, BLOCK_REG_K> desc_a1{ key_cache + offset, KEY_LINES_PER_LOAD - 1, (uint)(K * sizeof(half) - 1), (uint)(K * sizeof(half) - 1),
        0, 0 };
    // M[32:48]xK[0:32]
    offset = block_indices_p[block_idx + 2] * (HK * KV_BLOCK_SIZE * HEAD_SIZE * (uint)sizeof(half));
    lsc::block_2d_desc<half, 2, KEY_LINES_PER_LOAD, BLOCK_REG_K> desc_a2{ key_cache + offset, KEY_LINES_PER_LOAD - 1, (uint)(K * sizeof(half) - 1), (uint)(K * sizeof(half) - 1),
        0, 0 };
    // M[48:64]xK[0:32]
    offset = block_indices_p[block_idx + 3] * (HK * KV_BLOCK_SIZE * HEAD_SIZE * (uint)sizeof(half));
    lsc::block_2d_desc<half, 2, KEY_LINES_PER_LOAD, BLOCK_REG_K> desc_a3{ key_cache + offset, KEY_LINES_PER_LOAD - 1, (uint)(K * sizeof(half) - 1), (uint)(K * sizeof(half) - 1),
        0, 0 };
    // prefetch A
    block_idx = (uint)(id_wg_m * BLOCK_WG_M + id_sg_mn * (BLOCK_WG_M / SG_MN)) * STRIDE / KV_BLOCK_SIZE;
    offset = block_indices_p[block_idx] * (HK * KV_BLOCK_SIZE * HEAD_SIZE * (uint)sizeof(half));
    static_assert(BLOCK_WG_M / SG_MN <= KEY_LINES_PER_LOAD, "prefetch lines should be inside one block");
    lsc::block_2d_desc<half, 1, BLOCK_WG_M / SG_MN, 32> desc_prefetch_a{ key_cache + offset, BLOCK_WG_M / SG_MN - 1, (uint)(K * sizeof(half) - 1), (uint)(K * sizeof(half) - 1),
        0, 0 };
    // 0~2 M[:]xK[0:16] 2~4 K[16:32]                                                     --> 32 * 2 regs
    matrix<half, 4, BLOCK_REG_A> a0, a1, a2, a3;

    // warmup
    // prefetch
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
    desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
    desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);

    // load b: N[0:16]xK[0:16]
    cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0.row(0).format<int>(), desc_b0);
    cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b0.row(1).format<int>(), desc_b0);
    desc_b0.set_block_x(desc_b0.get_block_x() + 8);
    cm_sbarrier(1);

    auto dot = [&](matrix_ref<half, 2, BLOCK_REG_A> A0, matrix_ref<half, 2, BLOCK_REG_A> A1, matrix_ref<half, 2, BLOCK_REG_A> A2, matrix_ref<half, 2, BLOCK_REG_A> A3, matrix_ref<half, REG_N, BLOCK_REG_B> B) {
#pragma unroll
        for (int reg_n = 0; reg_n < REG_N; reg_n++) {
#pragma unroll
            for (uint reg_m = 0; reg_m < 2; reg_m++) {
                acc.row((ushort)(reg_m * REG_N + reg_n)) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(acc.row((ushort)(reg_m * REG_N + reg_n)),
                    B.row((ushort)reg_n).format<int>(), A0.row((ushort)reg_m).format<int>());
            }
#pragma unroll
            for (uint reg_m = 0; reg_m < 2; reg_m++) {
                acc.row((ushort)((reg_m + 2) * REG_N + reg_n)) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(acc.row((ushort)((reg_m + 2) * REG_N + reg_n)),
                    B.row((ushort)reg_n).format<int>(), A1.row((ushort)reg_m).format<int>());
            }
#pragma unroll
            for (uint reg_m = 0; reg_m < 2; reg_m++) {
                acc.row((ushort)((reg_m + 4) * REG_N + reg_n)) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(acc.row((ushort)((reg_m + 4) * REG_N + reg_n)),
                    B.row((ushort)reg_n).format<int>(), A2.row((ushort)reg_m).format<int>());
            }
#pragma unroll
            for (uint reg_m = 0; reg_m < 2; reg_m++) {
                acc.row((ushort)((reg_m + 6) * REG_N + reg_n)) = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(acc.row((ushort)((reg_m + 6) * REG_N + reg_n)),
                    B.row((ushort)reg_n).format<int>(), A3.row((ushort)reg_m).format<int>());
            }
        }
    };

    for (uint s = 0; s < STRIDE; s++) {
        #pragma unroll
        for (uint hs = 0; hs < HEAD_SIZE / BLOCK_WG_K; hs++) {
            // prefetch
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
            desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);
            if (hs == HEAD_SIZE / BLOCK_WG_K - 1)
                desc_prefetch_b.set_block_x((STRIDE - 1 - s - 1) * b_adjacent_between_head);
            else
                desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);

            // load b: N[0:16*2]xK[16:32]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0,  0>(b1.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b1.row(1).format<int>(), desc_b0);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);

            // load a: M[0:16*4]xK[0:32]
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a0.format<half>(), desc_a0);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a1.format<half>(), desc_a1);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a2.format<half>(), desc_a2);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a3.format<half>(), desc_a3);

            desc_a0.set_block_x(desc_a0.get_block_x() + 32);
            desc_a1.set_block_x(desc_a1.get_block_x() + 32);
            desc_a2.set_block_x(desc_a2.get_block_x() + 32);
            desc_a3.set_block_x(desc_a3.get_block_x() + 32);

            dot(a0.select<2, 1, BLOCK_REG_A, 1>(), a1.select<2, 1, BLOCK_REG_A, 1>(),
                a2.select<2, 1, BLOCK_REG_A, 1>(), a3.select<2, 1, BLOCK_REG_A, 1>(),
	    	    b0);

            // load b: N[0:16*2]xK[32:48]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0,  0>(b0.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b0.row(1).format<int>(), desc_b0);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);

            dot(a0.select<2, 1, BLOCK_REG_A, 1>(2), a1.select<2, 1, BLOCK_REG_A, 1>(2),
                a2.select<2, 1, BLOCK_REG_A, 1>(2), a3.select<2, 1, BLOCK_REG_A, 1>(2),
	            b1);

            // prefetch
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
            desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);
            desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);

            // load b: N[0:16*2]xK[48:64]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0,  0>(b1.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b1.row(1).format<int>(), desc_b0);
            if (hs == HEAD_SIZE / BLOCK_WG_K - 1)
                desc_b0.set_block_x((STRIDE - 1 - s - 1) * b_adjacent_between_head / 2);
            else
                desc_b0.set_block_x(desc_b0.get_block_x() + 8);
            // load a: M[0:32]xK[32:64]
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a0.format<half>(), desc_a0);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a1.format<half>(), desc_a1);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a2.format<half>(), desc_a2);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached>(a3.format<half>(), desc_a3);
            desc_a0.set_block_x(desc_a0.get_block_x() + 32);
            desc_a1.set_block_x(desc_a1.get_block_x() + 32);
            desc_a2.set_block_x(desc_a2.get_block_x() + 32);
            desc_a3.set_block_x(desc_a3.get_block_x() + 32);

            dot(a0.select<2, 1, BLOCK_REG_A, 1>(), a1.select<2, 1, BLOCK_REG_A, 1>(),
                a2.select<2, 1, BLOCK_REG_A, 1>(), a3.select<2, 1, BLOCK_REG_A, 1>(),
	    	    b0);

            // load b: N[0:16*4]xK[0:16]
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0,  0>(b0.row(0).format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached, 0, 16>(b0.row(1).format<int>(), desc_b0);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);

            dot(a0.select<2, 1, BLOCK_REG_A, 1>(2), a1.select<2, 1, BLOCK_REG_A, 1>(2),
                a2.select<2, 1, BLOCK_REG_A, 1>(2), a3.select<2, 1, BLOCK_REG_A, 1>(2),
	            b1);
            cm_sbarrier(0);
            cm_sbarrier(1);
        }
    }

    cm_sbarrier(0);

    matrix<half, REG_M * BLOCK_REG_M, REG_N * BLOCK_REG_N> acc_half;
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

    // if N(aka query) has tails, the following will not change the accuracy:
    //    gemm will compute results for the padding N(all should be zeros), the kq_max/kq_max_wg/kq_exp_partial_sum are along the query dimension and
    //    the results can be dropped in the future stage. To simplify the logic, the size of kq_max/kq_max_wg/kq_exp_partial_sum must be enough to hold
    //    all tails + padding results.
    int m_start = (int)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M);
    m_start = MYMIN(m_start, M);
    int m_end = MYMIN(m_start + BLOCK_SG_M, M);
    int valid_m = m_end - m_start;
    matrix<half, 32, 8> sum_t;
    vector<int, BLOCK_SG_N> seq;
    cmtl::cm_vector_assign(seq.select_all(), 0, 1);
#if IS_CAUSAL == 1
    bool skip_mask = false;
    // in streaming scenario, the past kvcache length may be arbitrary so valid causal mask of a workgroup may start at arbitrary position
    if (m_end <= (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N + q_start_strided)) {
        // all are inside causal mask == 1
        skip_mask = true;
    } else {
        vector<uint, BLOCK_SG_N> n_pos = (uint)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N + q_start_strided) + seq;
    #pragma unroll
        for (uint reg_m = 0; reg_m < REG_M * BLOCK_REG_M; reg_m++) {
            SIMD_IF_BEGIN (m_start + reg_m > n_pos) {
                acc_half.row(reg_m) = half{-60000};
            } SIMD_IF_END;
        }
    }
#else
    bool skip_mask = true;
#endif
    // case for valid_m == BLOCK_SG_M but skip_mask == false which needs to handle causal mask:
    //  query = 128 * 2 + 1, key = 256 * 2
    if (valid_m == BLOCK_SG_M && skip_mask) {
        vector<half, BLOCK_SG_N> max_n = acc_half.row(0);
    #pragma unroll
        for (uint reg_m = 1; reg_m < REG_M * BLOCK_REG_M; reg_m++) {
            max_n = cm_max<half>(max_n, acc_half.row(reg_m));
        }

        {
            uint slm_offset = (id_sg_m * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) * (uint)sizeof(half);
            // current max -> slm
            cm_slm_block_write(slm, slm_offset, max_n.format<int>());
            cm_slm_fence(CM_LOCAL_BARRIER);
            cm_barrier();
            // max inside wg
            cm_slm_block_read(slm, id_sg_n * BLOCK_SG_N * (uint)sizeof(half), max_n.format<int>());
            vector<half, BLOCK_SG_N> tmp;
    #pragma unroll
            for (uint i = 1; i < SG_M; i++) {
                slm_offset = (i * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) * (uint)sizeof(half);
                cm_slm_block_read(slm, slm_offset, tmp.format<int>());
                max_n = cm_max<half>(max_n, tmp);
            }
            // max across wg
            // kq_max: [b, hq, N_aligned]
            vector<uint, BLOCK_SG_N> max_offsets = (id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N + seq) * (uint)sizeof(half);
            cm_ptr_atomic<AtomicOp::FMAX, half>((half*)kq_max, max_offsets, max_n);
            
            // current max -> mem
            // kq_max_wg: [b, hq, M/BLOCK_WG_M, N_aligned]
            uint offset = (id_wg_m * N_aligned + id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) * sizeof(half);
            cm_ptr_store<int>((int*)kq_max_wg, offset, max_n.format<int>());
        }
        {
            // kq_exp_partial_sum: [b, hq, N_aligned, M/(BLOCK_SIZE/STRIDE)]
            matrix<half, 8, 32> sum;
    #pragma unroll
            for (uint m = 0; m < BLOCK_SG_M / block_size_div_stride; m++) {
                sum.row(m) = cm_exp((acc_half.row(m * block_size_div_stride) - max_n) * log2e);
    #pragma unroll
                for (uint sub_m = 1; sub_m < block_size_div_stride; sub_m++) {
                    uint real_m = m * block_size_div_stride + sub_m;
                    sum.row(m) += cm_exp((acc_half.row(real_m) - max_n) * log2e);
                }
            }

            Transpose_8x32(sum, sum_t);
        }
    } else {
        // M tails
        vector<half, BLOCK_SG_N> max_n = -60000;
        for (uint reg_m = 0; reg_m < valid_m; reg_m++) {
            max_n = cm_max<half>(max_n, acc_half.row(reg_m));
        }

        {
            uint slm_offset = (id_sg_m * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) * (uint)sizeof(half);
            // current max -> slm
            cm_slm_block_write(slm, slm_offset, max_n.format<int>());
            cm_slm_fence(CM_LOCAL_BARRIER);
            cm_barrier();
            // max inside wg
            cm_slm_block_read(slm, id_sg_n * BLOCK_SG_N * (uint)sizeof(half), max_n.format<int>());
            vector<half, BLOCK_SG_N> tmp;
    #pragma unroll
            for (uint i = 1; i < SG_M; i++) {
                slm_offset = (i * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) * (uint)sizeof(half);
                cm_slm_block_read(slm, slm_offset, tmp.format<int>());
                max_n = cm_max<half>(max_n, tmp);
            }
            // max across wg
            // kq_max: [b, hq, N_aligned]
            vector<uint, BLOCK_SG_N> max_offsets = (id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N + seq) * (uint)sizeof(half);
            cm_ptr_atomic<AtomicOp::FMAX, half>((half*)kq_max, max_offsets, max_n);

            // current max -> mem
            // kq_max_wg: [b, hq, M/BLOCK_WG_M, N_aligned]
            uint offset = (id_wg_m * N_aligned + id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) * sizeof(half);
            cm_ptr_store<int>((int*)kq_max_wg, offset, max_n.format<int>());
        }
        {
            // kq_exp_partial_sum: [b, hq, N_aligned, M/(BLOCK_SIZE/STRIDE)]
            matrix<half, 8, 32> sum = 0;
            vector<uint, BLOCK_SG_N> n_pos = (uint)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N + q_start_strided) + seq;
    #pragma unroll
            for (uint m = 0; m < BLOCK_SG_M / block_size_div_stride; m++) {
    #pragma unroll
                for (uint sub_m = 0; sub_m < block_size_div_stride; sub_m++) {
                    uint real_m = m * block_size_div_stride + sub_m;
#if IS_CAUSAL == 1
                    // to following case:
                    // 0 0 1 1
                    // 0 0 0 1
                    // the acc value of first column should be -inf --> max(first column) == -inf --> exp(first column - max) == 1, this is incorrect
                    // so need to use simd_if to detect per element state
                    SIMD_IF_BEGIN ((m_start + real_m <= n_pos) & (real_m < valid_m)) {
                        sum.row(m) += cm_exp((acc_half.row(real_m) - max_n) * log2e);
                    } SIMD_IF_END;
#else
                    if (real_m < valid_m)
                        sum.row(m) += cm_exp((acc_half.row(real_m) - max_n) * log2e);
#endif
                }
            }
            Transpose_8x32(sum, sum_t);
        }
    }
    // store
    lsc::block_2d_desc<half, 1, 8, 8> desc_c{ kq_exp_partial_sum, N - 1, (uint)(M_block_aligned * sizeof(half) - 1), (uint)(M_block_aligned * sizeof(half) - 1),
        (int)((id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) / block_size_div_stride), (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) };
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 0>(desc_c, sum_t.select<8, 1, 8, 1>( 0).format<half>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 1>(desc_c, sum_t.select<8, 1, 8, 1>( 8).format<half>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 2>(desc_c, sum_t.select<8, 1, 8, 1>(16).format<half>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 3>(desc_c, sum_t.select<8, 1, 8, 1>(24).format<half>());
}
#endif

#if 1 || (BLOCK_SG_M == 64 && BLOCK_SG_N == 32)
// const static int channels_reduce_32[] = { 0, 16,  8, 24,  4, 20, 12, 28,  2, 18, 10, 26,  6, 22, 14, 30,  
// 		                                  1, 17,  9, 25,  5, 21, 13, 29,  3, 19, 11, 27,  7, 23, 15, 31};
// src_a is query, src_b is key
CM_INLINE void gemm_qk_64x32_xe2(uint id_wg_m, uint id_wg_n, uint hq, uint slm, svmptr_t key_cache ATTR, svmptr_t query ATTR, svmptr_t block_indices ATTR, svmptr_t block_indices_begins ATTR, svmptr_t kq_max_wg ATTR, svmptr_t kq_exp_partial_sum ATTR,
uint M, uint N, uint K, uint query_stride, uint q_start_strided) {
    constexpr int SG_SIZE = 16;
    constexpr int BLOCK_WG_K = 64;	// same in sg
#ifndef BLOCK_SG_M
    #define BLOCK_SG_M  64
    #define BLOCK_SG_N  32
    #define SG_M  4
    #define SG_N  4
    #define HEAD_SIZE  128
    #define KV_BLOCK_SIZE  256
    #define STRIDE  16
#endif
    // xehpg DPAS spec: dst: [8, 8], repeat: 1~8, depth: 8
    static constexpr int REPEAT = 8;
    static constexpr int DEPTH = 8;
    static constexpr int BLOCK_REG_M = REPEAT;
    static constexpr int BLOCK_REG_N = SG_SIZE;
    static constexpr int BLOCK_DPAS_C = BLOCK_REG_M * BLOCK_REG_N;
    static constexpr int VNNI = sizeof(half);
    static constexpr int BLOCK_REG_K = DEPTH * sizeof(int) / VNNI;
    static constexpr int BLOCK_REG_A = BLOCK_REG_M * BLOCK_REG_K;
    static constexpr int BLOCK_REG_B = BLOCK_REG_N * BLOCK_REG_K;
    static constexpr int BLOCK_WG_M = SG_M * BLOCK_SG_M;
    static constexpr int BLOCK_WG_N = SG_N * BLOCK_SG_N;
    // register blocking
    static constexpr int REG_M = BLOCK_SG_M / BLOCK_REG_M;
    static constexpr int REG_N = BLOCK_SG_N / BLOCK_REG_N;
    static constexpr int REG_K = BLOCK_WG_K / BLOCK_REG_K;
    static constexpr int REG_MN = REG_M * REG_N;
    static constexpr int KEY_LINES_PER_LOAD = KV_BLOCK_SIZE / STRIDE;

    matrix<float, REG_M * REG_N, BLOCK_DPAS_C> acc = 0;                              // --> 64*2 regs
    uint id_sg_n = cm_local_id(0);
    uint id_sg_m = cm_local_id(1);
    uint id_sg_mn = id_sg_m * SG_N + id_sg_n;

    static_assert(REG_N == 2, "block_2d_desc for b is manually unrolled by 2");
    static_assert(HEAD_SIZE % BLOCK_WG_K == 0, "K dimension must be multiple of BLOCK_WG_K");
    static_assert(KV_BLOCK_SIZE == 256, "block size of key(key_cache) should be 256");
    uint N_block = (N + BLOCK_WG_N - 1) / BLOCK_WG_N;
    uint M_aligned = (M + BLOCK_WG_M - 1) / BLOCK_WG_M * BLOCK_WG_M;
    uint K_block_pad = N_block * (BLOCK_WG_N / (BLOCK_SIZE / STRIDE));
    const uint block_size_div_stride = BLOCK_SIZE / STRIDE;
    constexpr SOFTMAX_TYPE log2e = 1.4426950408889634f;
    //static_assert(BLOCK_SG_M / block_size_div_stride == 8, "BLOCK_SG_M / block_size_div_stride should be 8");

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
        }

        return;
    }
#endif
    // assume block index coming from 0 in block_indices_begins
    int block_index_begin = ((int*)block_indices_begins)[0];
    int* block_indices_p = (int*)block_indices + block_index_begin;
    int b_adjacent_between_head = query_stride / STRIDE;
    // M[0:16*2]xK[0:16]
    lsc::block_2d_desc<half, 1, 32, BLOCK_REG_K> desc_a{ query, M - 1, (uint)((query_stride - hq * HEAD_SIZE) * sizeof(half) - 1), (uint)(query_stride * sizeof(half) - 1),
        (STRIDE - 1) * b_adjacent_between_head, (int)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) };
    // prefetch A
    static constexpr int SG_MN = SG_M * SG_N;
    lsc::block_2d_desc<half, 1, BLOCK_WG_M / SG_MN, 32> desc_prefetch_a{query, M - 1, (uint)((query_stride - hq * HEAD_SIZE) * sizeof(half) - 1), (uint)(query_stride * sizeof(half) - 1),
        (STRIDE - 1) * b_adjacent_between_head, (int)(id_wg_m * BLOCK_WG_M + id_sg_mn * (BLOCK_WG_M / SG_MN)) };
    // M[0:16*2]xK[0:16]                                                                  --> 8+8 regs
    matrix<half, REG_M, BLOCK_REG_A> a0;

    // M[0:16]xK[0:32]
    uint block_idx = (uint)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) * STRIDE / KV_BLOCK_SIZE;
    uint max_block_idx = (uint)(N * STRIDE + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE - 1;
    block_idx = MYMIN(block_idx, max_block_idx);
#if USE_INT8
    uint offset = block_indices_p[block_idx] * (HK * KV_BLOCK_SIZE * HEAD_SIZE_KEY * (uint)sizeof(char));
    lsc::block_2d_desc<int, 1, KEY_LINES_PER_LOAD, 8> desc_b0{ key_cache + offset, KEY_LINES_PER_LOAD - 1, (uint)(K * sizeof(char) - 1), (uint)(K * sizeof(char) - 1),
        0, 0 };
    uint scale_offset0 = offset + KV_BLOCK_SIZE * HEAD_SIZE;
    block_idx = MYMIN(block_idx + 1, max_block_idx);
    offset = block_indices_p[block_idx] * (HK * KV_BLOCK_SIZE * HEAD_SIZE_KEY * (uint)sizeof(char));
    lsc::block_2d_desc<int, 1, KEY_LINES_PER_LOAD, 8> desc_b1{ key_cache + offset, KEY_LINES_PER_LOAD - 1, (uint)(K * sizeof(char) - 1), (uint)(K * sizeof(char) - 1),
        0, 0 };
    uint scale_offset1 = offset + KV_BLOCK_SIZE * HEAD_SIZE;
    // prefetch B
    block_idx = (uint)(id_wg_n * BLOCK_WG_N + id_sg_mn * (BLOCK_WG_N / SG_MN)) * STRIDE / KV_BLOCK_SIZE;
    block_idx = MYMIN(block_idx, max_block_idx);
    offset = block_indices_p[block_idx] * (HK * KV_BLOCK_SIZE * HEAD_SIZE_KEY * (uint)sizeof(char));
    static_assert(BLOCK_WG_N / SG_MN <= KEY_LINES_PER_LOAD, "prefetch lines should be inside one block");
    lsc::block_2d_desc<uchar, 1, BLOCK_WG_N / SG_MN, 32> desc_prefetch_b{ key_cache + offset, BLOCK_WG_N / SG_MN - 1, (uint)(K * sizeof(char) - 1), (uint)(K * sizeof(char) - 1),
        0, 0 };

    // N[:]xK[0:32]                                                     --> 16 * 1 regs
    matrix<int, KEY_LINES_PER_LOAD, 8> b0_up_s8, b0_down_s8, b1_up_s8, b1_down_s8;
    matrix<half, 2, BLOCK_REG_B> b0;                      // --> 16 regs
    matrix<half, 2, KEY_LINES_PER_LOAD * 2> scales, zps;
    matrix<half, 2, KV_BLOCK_SIZE> scales_block, zps_block;
#else
    uint offset = block_indices_p[block_idx] * (HK * KV_BLOCK_SIZE * HEAD_SIZE * (uint)sizeof(half));
    lsc::block_2d_desc<int, 1, KEY_LINES_PER_LOAD, 8> desc_b0{ key_cache + offset, KEY_LINES_PER_LOAD - 1, (uint)(K * sizeof(half) - 1), (uint)(K * sizeof(half) - 1),
        0, 0 };
    // printf("===============0 lid:%d.%d, block_idx=%d, offset=%u\n", id_sg_n, id_sg_m, block_idx, offset);
    block_idx = MYMIN(block_idx + 1, max_block_idx);
    offset = block_indices_p[block_idx] * (HK * KV_BLOCK_SIZE * HEAD_SIZE * (uint)sizeof(half));
    lsc::block_2d_desc<int, 1, KEY_LINES_PER_LOAD, 8> desc_b1{ key_cache + offset, KEY_LINES_PER_LOAD - 1, (uint)(K * sizeof(half) - 1), (uint)(K * sizeof(half) - 1),
        0, 0 };
    // printf("===============1 lid:%d.%d, block_idx=%d, offset=%u\n", id_sg_n, id_sg_m, block_idx, offset);
    // prefetch B
    block_idx = (uint)(id_wg_n * BLOCK_WG_N + id_sg_mn * (BLOCK_WG_N / SG_MN)) * STRIDE / KV_BLOCK_SIZE;
    block_idx = MYMIN(block_idx, max_block_idx);
    offset = block_indices_p[block_idx] * (HK * KV_BLOCK_SIZE * HEAD_SIZE * (uint)sizeof(half));
    static_assert(BLOCK_WG_N / SG_MN <= KEY_LINES_PER_LOAD, "prefetch lines should be inside one block");
    lsc::block_2d_desc<half, 1, BLOCK_WG_N / SG_MN, 32> desc_prefetch_b{ key_cache + offset, BLOCK_WG_N / SG_MN - 1, (uint)(K * sizeof(half) - 1), (uint)(K * sizeof(half) - 1),
        0, 0 };
    // printf("===============2 lid:%d.%d, block_idx=%d, offset=%u\n", id_sg_n, id_sg_m, block_idx, offset);
    // 0~2 M[:]xK[0:16] 2~4 K[16:32]                                                     --> 32 * 2 regs
    matrix<half, 2, BLOCK_REG_B> b0, b1;
#endif

    // warmup
    // prefetch
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
    desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);
    cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
    desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);

    // load b: N[0:16]xK[0:16]
#if USE_INT8
    {
        lsc::block_2d_desc<int, 1, 16, 16 / 2> desc_scale{ key_cache + scale_offset0, 16 * 2 - 1, (uint)(16 * sizeof(half) - 1), (uint)(16 * sizeof(half) - 1),
            0, 0 };
        matrix<half, 16, 16> tmp_scale, tmp_zp;
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
    }

    cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0_up_s8.format<int>(), desc_b0);
    cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0_down_s8.format<int>(), desc_b1);
#else
    cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0[0].format<int>(), desc_b0);
    cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0[1].format<int>(), desc_b1);
#endif
    desc_b0.set_block_x(desc_b0.get_block_x() + 8);
    desc_b1.set_block_x(desc_b1.get_block_x() + 8);

    cm_sbarrier(1);

#if USE_INT8
    auto dec = [&](vector<int, 64> B0_i8, vector<int, 64> B1_i8, matrix_ref<half, REG_N, BLOCK_REG_B> B0) {
#pragma unroll
        for (int n = 0; n < REG_N; n++) {
            auto b = B0[n].format<half, 8, 32>();
#pragma unroll
            for (int m = 0; m < 8; m++) {
                auto b_row = b[m];
                vector<ushort, 16> d0;
                if (n == 0)
                    d0 = B0_i8.format<ushort, 4, 32>()[m / 2].select<16, 2>(m % 2);
                else
                    d0 = B1_i8.format<ushort, 4, 32>()[m / 2].select<16, 2>(m % 2);
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
        auto tmp = scales_block[0].select<16, 1>(s * 16);
        scales[0].select<16, 2>(0) = tmp;
        scales[0].select<16, 2>(1) = scales[0].select<16, 2>(0);
        tmp = scales_block[1].select<16, 1>(s * 16);
        scales[1].select<16, 2>(0) = tmp;
        scales[1].select<16, 2>(1) = scales[1].select<16, 2>(0);
        tmp = zps_block[0].select<16, 1>(s * 16);
        zps[0].select<16, 2>(0) = tmp;
        zps[0].select<16, 2>(1) = zps[0].select<16, 2>(0);
        tmp = zps_block[1].select<16, 1>(s * 16);
        zps[1].select<16, 2>(0) = tmp;
        zps[1].select<16, 2>(1) = zps[1].select<16, 2>(0);
#endif
        #pragma unroll
        for (uint hs = 0; hs < HEAD_SIZE / BLOCK_WG_K; hs++) {
            // prefetch
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
            desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);
            if (hs == HEAD_SIZE / BLOCK_WG_K - 1)
                desc_prefetch_a.set_block_x((STRIDE - 1 - s - 1) * b_adjacent_between_head);
            else
                desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);

            // load a: M[0:16*4]xK[0:16]
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0,  0>(a0.select<4, 1, BLOCK_REG_A, 1>(0).format<half>(), desc_a);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0, 32>(a0.select<4, 1, BLOCK_REG_A, 1>(4).format<half>(), desc_a);
            // load b: N[0:16*2]xK[16:32]
#if USE_INT8
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b1_up_s8.format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b1_down_s8.format<int>(), desc_b1);
            dec(b0_up_s8.format<int>().select<64, 1>(), b0_down_s8.format<int>().select<64, 1>(), b0);
            dot(a0, b0);
#else
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b1[0].format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b1[1].format<int>(), desc_b1);
            dot(a0, b0);
#endif

            desc_a.set_block_x(desc_a.get_block_x() + 16);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);
            desc_b1.set_block_x(desc_b1.get_block_x() + 8);

            // load a: M[0:16*4]xK[16:32]
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0,  0>(a0.select<4, 1, BLOCK_REG_A, 1>(0).format<half>(), desc_a);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0, 32>(a0.select<4, 1, BLOCK_REG_A, 1>(4).format<half>(), desc_a);

#if USE_INT8
            dec(b0_up_s8.format<int>().select<64, 1>(64), b0_down_s8.format<int>().select<64, 1>(64), b0);
            dot(a0, b0);
#else
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0[0].format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0[1].format<int>(), desc_b1);
            dot(a0, b1);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);
            desc_b1.set_block_x(desc_b1.get_block_x() + 8);
#endif
            desc_a.set_block_x(desc_a.get_block_x() + 16);

            // prefetch
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_b);
            cm_prefetch<CacheHint::Cached, CacheHint::Cached>(desc_prefetch_a);
            desc_prefetch_b.set_block_x(desc_prefetch_b.get_block_x() + 32);
            desc_prefetch_a.set_block_x(desc_prefetch_a.get_block_x() + 32);

            // load a: M[0:16*4]xK[32:48]
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0,  0>(a0.select<4, 1, BLOCK_REG_A, 1>(0).format<half>(), desc_a);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0, 32>(a0.select<4, 1, BLOCK_REG_A, 1>(4).format<half>(), desc_a);

            // load b: N[0:16*2]xK[32:64]
#if USE_INT8
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0_up_s8.format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0_down_s8.format<int>(), desc_b1);
            dec(b1_up_s8.format<int>().select<64, 1>(), b1_down_s8.format<int>().select<64, 1>(), b0);
            dot(a0, b0);
#else
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b1[0].format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b1[1].format<int>(), desc_b1);
            dot(a0, b0);
#endif

            desc_a.set_block_x(desc_a.get_block_x() + 16);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);
            desc_b1.set_block_x(desc_b1.get_block_x() + 8);

            // load a: M[0:16*4]xK[48:64]
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0,  0>(a0.select<4, 1, BLOCK_REG_A, 1>(0).format<half>(), desc_a);
            cm_load<lsc::Normal, CacheHint::Cached, CacheHint::Cached, 0, 32>(a0.select<4, 1, BLOCK_REG_A, 1>(4).format<half>(), desc_a);
            if (hs == HEAD_SIZE / BLOCK_WG_K - 1) {
                desc_a.set_block_x((STRIDE - 1 - s - 1) * b_adjacent_between_head);
            } else {
                desc_a.set_block_x(desc_a.get_block_x() + 16);
            }

#if USE_INT8
            dec(b1_up_s8.format<int>().select<64, 1>(64), b1_down_s8.format<int>().select<64, 1>(64), b0);
            dot(a0, b0);
#else
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0[0].format<int>(), desc_b0);
            cm_load<lsc::Transpose, CacheHint::Cached, CacheHint::Cached>(b0[1].format<int>(), desc_b1);
            desc_b0.set_block_x(desc_b0.get_block_x() + 8);
            desc_b1.set_block_x(desc_b1.get_block_x() + 8);
            dot(a0, b1);
#endif

            cm_sbarrier(0);
            cm_sbarrier(1);
        }
    }

    cm_sbarrier(0);

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

    // if M(aka query) has tails, the following will not change the accuracy:
    //    gemm will compute results for the padding M(all should be zeros), the kq_max/kq_max_wg/kq_exp_partial_sum are along the query dimension and
    //    the results can be dropped in the future stage. To simplify the logic, the size of kq_max/kq_max_wg/kq_exp_partial_sum must be enough to hold
    //    all tails + padding results.
    int n_start = (int)(id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N);
    n_start = MYMIN(n_start, N);
    int n_end = MYMIN(n_start + BLOCK_SG_N, N);
    int valid_n = n_end - n_start;
    matrix<SOFTMAX_TYPE, 64, 4> sum_t;
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
    max_m.select<32, 1>() = reduce2d<1, 0, 1>(acc_half.select<32, 1, 32, 1>()).format<SOFTMAX_TYPE>();
    max_m.select<32, 1>(32) = reduce2d<1, 0, 1>(acc_half.select<32, 1, 32, 1>(32)).format<SOFTMAX_TYPE>();

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
        sum_t.select<32, 1, 4, 1>( 0).format<SOFTMAX_TYPE>() = reduce2d<4, 1, 4>(acc_half.select<32, 1, 32, 1>( 0)).format<SOFTMAX_TYPE>();
        sum_t.select<32, 1, 4, 1>(32).format<SOFTMAX_TYPE>() = reduce2d<4, 1, 4>(acc_half.select<32, 1, 32, 1>(32)).format<SOFTMAX_TYPE>();
    }
    // store
    lsc::block_2d_desc<SOFTMAX_TYPE, 1, 8, 4> desc_c{ kq_exp_partial_sum, M - 1, (uint)(K_block_pad * sizeof(SOFTMAX_TYPE) - 1), (uint)(K_block_pad * sizeof(SOFTMAX_TYPE) - 1),
        (int)((id_wg_n * BLOCK_WG_N + id_sg_n * BLOCK_SG_N) / block_size_div_stride), (int)(id_wg_m * BLOCK_WG_M + id_sg_m * BLOCK_SG_M) };
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 0>(desc_c, sum_t.select<8, 1, 4, 1>( 0).format<SOFTMAX_TYPE>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 1>(desc_c, sum_t.select<8, 1, 4, 1>( 8).format<SOFTMAX_TYPE>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 2>(desc_c, sum_t.select<8, 1, 4, 1>(16).format<SOFTMAX_TYPE>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 3>(desc_c, sum_t.select<8, 1, 4, 1>(24).format<SOFTMAX_TYPE>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 4>(desc_c, sum_t.select<8, 1, 4, 1>(32).format<SOFTMAX_TYPE>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 5>(desc_c, sum_t.select<8, 1, 4, 1>(40).format<SOFTMAX_TYPE>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 6>(desc_c, sum_t.select<8, 1, 4, 1>(48).format<SOFTMAX_TYPE>());
    cm_store<CacheHint::Uncached, CacheHint::WriteBack, 0, 8 * 7>(desc_c, sum_t.select<8, 1, 4, 1>(56).format<SOFTMAX_TYPE>());
}
#endif
