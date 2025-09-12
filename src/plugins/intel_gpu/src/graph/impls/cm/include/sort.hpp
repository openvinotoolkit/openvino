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

#ifndef ATTR
#define ATTR [[type("svmptr_t")]]
#define ATTR_BUF [[type("buffer_t")]]
#endif

#if 0
template<typename T, int N>
void show(const vector<T, N> mat, bool is_hex=true) {
    printf("vector [%d]:\n[", N);
    for(int n = 0; n < N; n ++) {
        if (is_hex)
            printf("%x,", mat[n]);
        else
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
#endif

// https://gpuopen.com/download/Introduction_to_GPU_Radix_Sort.pdf
template<typename TYPE>
CM_INLINE void sort(uint slm, svmptr_t src, svmptr_t sorted_value, svmptr_t sorted_index, svmptr_t sort_tmp, uint n) {
    const ushort THREADS = 32;
    vector<unsigned, THREADS> seq_u32;
    vector<ushort, THREADS> seq;
    cmtl::cm_vector_assign(seq.select_all(), 0, 1);
    cmtl::cm_vector_assign(seq_u32.select_all(), 0, 1);
    svmptr_t sorted_src = src;
    svmptr_t sorted_tmp = sorted_value;
    svmptr_t cur_idx = sorted_index;
    svmptr_t cur_idx_tmp = sort_tmp;
    uint iter = (n + THREADS - 1) / THREADS;
    vector<unsigned, THREADS> offset_src;
    cmtl::cm_vector_assign(offset_src.select_all(), 0, iter);
    auto f16_u16 = [] (vector_ref<half, THREADS> in, vector_ref<ushort, THREADS> out) {
        static const ushort HIGH_BIT = 1 << 15;
        auto in_u16 = in.format<ushort>();
        auto mask = (in_u16 & HIGH_BIT) != 0;
        vector<ushort, THREADS> m;
        m.merge(ushort{0xffff}, HIGH_BIT, mask);
        out = in_u16 ^ m;
    };
    auto u16_f16 = [] (vector_ref<ushort, THREADS> in, vector_ref<half, THREADS> out) {
        static const ushort HIGH_BIT = 1 << 15;
        auto mask = (in & HIGH_BIT) != 0;
        vector<ushort, THREADS> m;
        m.merge(HIGH_BIT, ushort{0xffff}, mask);
        out.format<ushort>() = in ^ m;
    };
    if constexpr(std::is_same<TYPE, half>::value) {
        {
            // f16 to u16
            vector<ushort, THREADS> data;
            vector<half, THREADS> data_f;
            int i;
            for (i = 0; i + THREADS <= n / THREADS * THREADS; i += THREADS) {
                data_f.format<int>() = cm_ptr_load<int, THREADS / 2>((int*)src, i * (int)sizeof(short));
                f16_u16(data_f, data);
                cm_ptr_store<int, THREADS / 2>((int*)src, i * (int)sizeof(short), data.format<int>());
            }
            if (i < n) {
                auto pos = seq_u32 + i;
                SIMD_IF_BEGIN (pos < n) {
                    data_f = cm_ptr_load<half>((half*)src, pos * (uint)sizeof(half));
                    f16_u16(data_f, data);
                    cm_ptr_store<ushort>((ushort*)src, pos * (uint)sizeof(ushort), data);
                } SIMD_IF_END;
            }
        }
    }
    {
        // generate idx
        vector<ushort, THREADS> data;
        int i;
        for (i = 0; i + THREADS <= n / THREADS * THREADS; i += THREADS) {
            data = seq + i;
            cm_ptr_store<int, THREADS / 2>((int*)cur_idx, i * (int)sizeof(short), data.format<int>());
        }
        if (i < n) {
            auto pos = seq_u32 + i;
            data = seq + i;

            SIMD_IF_BEGIN (pos < n) {
                cm_ptr_store<ushort>((ushort*)cur_idx, pos * (uint)sizeof(ushort), data);
            } SIMD_IF_END;
        }
    }
    // 4bit per pass, 4 pass for f16
    for (int pass = 0; pass < 4; pass++) {
        {
            // slm layout: short [16][work items] = 16*32*2 bytes
            vector<int, 64> data = 0;
            for (int i = 0; i < 16 * THREADS * sizeof(ushort); i += 256)
                cm_slm_block_write(slm, i, data);
        }
        {
            // counting phase
            vector<ushort, THREADS> data;
            for (int i = 0; i < iter; i++) {
                data = cm_ptr_load<ushort>((ushort*)sorted_src, (offset_src + i) * (uint)sizeof(ushort), (offset_src + i) < n);
                vector<ushort, THREADS> bits = 0xf - ((data >> (pass * 4)) & 0xf);
                vector<ushort, THREADS> addr = bits * THREADS + seq;
                vector<ushort, THREADS> total;
                cm_slm_read(slm, addr, total);
                total += 1;
                cm_slm_write(slm, addr, total);
            }
        }
        // {
        //     // prefix sum
        //     vector<ushort, 16 * THREADS> data;
        //     cm_slm_block_read(slm, 0, data);
        //     for (int i = 1; i < 16 * THREADS; i++) {
        //         data[i] += data[i - 1];
        //     }
        //     data.select<16 * THREADS - 1, 1>(1) = data.select<16 * THREADS - 1, 1>(0);
        //     data[0] = 0;
        //     cm_slm_block_write(slm, 0, data);
        // }
        {
            // prefix sum
            vector<ushort, 16> local_prefix = 0;

            vector<ushort, 16> seq_prefix;
            cmtl::cm_vector_assign(seq_prefix.select_all(), 0, THREADS);

            #pragma unroll
            for (ushort i = 0; i < THREADS; i++) {
                auto prev = local_prefix;
                vector<ushort, 16> hist;
                vector<ushort, 16> addr = seq_prefix + i;
                cm_slm_read(slm, addr, hist);
                local_prefix += hist;
                cm_slm_write(slm, addr, prev);
            }
            // Hillis-Steele scan
            vector<ushort, 16> local_tmp;
            local_tmp.select<15, 1>(1) = local_prefix.select<15, 1>(1) + local_prefix.select<15, 1>(0); local_tmp[0] = local_prefix[0];
            local_prefix.select<14, 1>(2) = local_tmp.select<14, 1>(2) + local_tmp.select<14, 1>(0); local_prefix.select<2, 1>(0) = local_tmp.select<2, 1>(0);
            local_tmp.select<12, 1>(4) = local_prefix.select<12, 1>(4) + local_prefix.select<12, 1>(0); local_tmp.select<4, 1>(0) = local_prefix.select<4, 1>(0);
            local_prefix.select<8, 1>(8) = local_tmp.select<8, 1>(8) + local_tmp.select<8, 1>(0); local_prefix.select<8, 1>(0) = local_tmp.select<8, 1>(0);
            vector<ushort, 16 * THREADS> data;
            cm_slm_block_read(slm, 0, data);
            #pragma unroll
            for (int i = 1; i < 16; i++) {
                data.select<THREADS, 1>(i * THREADS) += local_prefix[i - 1];
            }
            cm_slm_block_write(slm, 0, data);
        }
        {
            // reorder
            vector<ushort, THREADS> data;
            for (int i = 0; i < iter; i++) {
                data = cm_ptr_load((ushort*)sorted_src, (offset_src + i) * (uint)sizeof(ushort), (offset_src + i) < n);
                vector<ushort, THREADS> bits = 0xf - ((data >> (pass * 4)) & 0xf);
                vector<ushort, THREADS> addr = bits * THREADS + seq;
                vector<ushort, THREADS> index;
                cm_slm_read(slm, addr, index);
                vector<unsigned, THREADS> offset_i32 = index * (uint)sizeof(ushort);
                cm_ptr_store((ushort*)sorted_tmp, offset_i32, data, index < n);

                data = cm_ptr_load((ushort*)cur_idx, (offset_src + i) * (uint)sizeof(ushort));
                cm_ptr_store((ushort*)cur_idx_tmp, offset_i32, data, (offset_src + i) < n);

                index += 1;
                cm_slm_write(slm, addr, index);
            }
            auto tmp = sorted_src;
            sorted_src = sorted_tmp;
            sorted_tmp = tmp;
            tmp = cur_idx;
            cur_idx = cur_idx_tmp;
            cur_idx_tmp = tmp;
        }
    }
    {
        // copy to output
        vector<ushort, THREADS> data;
        vector<half, THREADS> data_f;
        int i;
        for (i = 0; i + THREADS <= n / THREADS * THREADS; i += THREADS) {
            data.format<int>() = cm_ptr_load<int, THREADS / 2>((int*)src, i * (int)sizeof(short));
            if constexpr(std::is_same<TYPE, half>::value) {
                u16_f16(data, data_f);
                cm_ptr_store<int, THREADS / 2>((int*)sorted_value, i * (int)sizeof(short), data_f.format<int>());
            } else {
                cm_ptr_store<int, THREADS / 2>((int*)sorted_value, i * (int)sizeof(short), data.format<int>());
            }
        }
        if (i < n) {
            auto pos = seq_u32 + i;
            if constexpr(std::is_same<TYPE, half>::value) {
                SIMD_IF_BEGIN (pos < n) {
                    data = cm_ptr_load((ushort*)src, pos * (uint)sizeof(half));
                    u16_f16(data, data_f);
                    cm_ptr_store<half>((half*)sorted_value, pos * (uint)sizeof(half), data_f);
                } SIMD_IF_END;
            } else {
                SIMD_IF_BEGIN (pos < n) {
                    data = cm_ptr_load((ushort*)src, pos * (uint)sizeof(ushort));
                    cm_ptr_store<ushort>((ushort*)sorted_value, pos * (uint)sizeof(ushort), data);
                } SIMD_IF_END;
            }
        }
    }
}
