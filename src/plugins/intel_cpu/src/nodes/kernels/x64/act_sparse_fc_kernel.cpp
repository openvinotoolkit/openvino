// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cstring>
#include "act_sparse_fc_kernel.hpp"

#include "openvino/core/parallel.hpp"

#include "/home/tingqian/aboutSHW/include/linux_perf.hpp"

#if defined(HAVE_AVX2)
#    include <immintrin.h>
#endif

// https://github.com/intel-sandbox/dynSparseFC/blob/main/dyn_sparse_fc.cpp

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

#if defined(HAVE_AVX2)
static __m256 load_fp32x8(float* p) {
    return _mm256_loadu_ps(p);
}
static __m256 load_fp32x8(ov::float16* p) {
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<__m128i const*>(p)));
}

/*
dst               : result
pw0/stride_w      : weight matrix in [IC, OC] layout
pscale            : input activations
gate_ids/gate_cnt : an array of input channels with non-zero activations (after thresholding)
*/
template <typename WType>
void _sparse_accumulate(float* dst,
                        int64_t OC,
                        WType* pw0,
                        int64_t stride_w,
                        float* pscale,
                        int64_t* gate_ids,
                        int gate_cnt) {
    int i = 0;
    int g = 0;
    for (; g + 3 < gate_cnt; g += 4) {
        auto row0 = gate_ids[g];
        auto row1 = gate_ids[g + 1];
        auto row2 = gate_ids[g + 2];
        auto row3 = gate_ids[g + 3];
        auto p_w0 = pw0 + row0 * stride_w;
        auto p_w1 = pw0 + row1 * stride_w;
        auto p_w2 = pw0 + row2 * stride_w;
        auto p_w3 = pw0 + row3 * stride_w;
        auto vscale0 = _mm256_broadcast_ss(pscale + row0);
        auto vscale1 = _mm256_broadcast_ss(pscale + row1);
        auto vscale2 = _mm256_broadcast_ss(pscale + row2);
        auto vscale3 = _mm256_broadcast_ss(pscale + row3);
        for (i = 0; i + 7 < OC; i += 8) {
            auto vdst = _mm256_loadu_ps(dst + i);
            auto vw0 = load_fp32x8(p_w0 + i);
            auto vw1 = load_fp32x8(p_w1 + i);
            auto vw2 = load_fp32x8(p_w2 + i);
            auto vw3 = load_fp32x8(p_w3 + i);
            // prefetch
            //_mm_prefetch(p_w0 + i + 1024, _MM_HINT_T1);
            //_mm_prefetch(p_w1 + i + 1024, _MM_HINT_T1);
            //_mm_prefetch(p_w2 + i + 1024, _MM_HINT_T1);
            //_mm_prefetch(p_w3 + i + 1024, _MM_HINT_T1);
            vdst = _mm256_fmadd_ps(vw0, vscale0, vdst);
            vdst = _mm256_fmadd_ps(vw1, vscale1, vdst);
            vdst = _mm256_fmadd_ps(vw2, vscale2, vdst);
            vdst = _mm256_fmadd_ps(vw3, vscale3, vdst);
            _mm256_storeu_ps(dst + i, vdst);
        }
    }

    for (; g < gate_cnt; g++) {
        auto row = gate_ids[g];
        auto pw = pw0 + row * stride_w;
        auto vscale = _mm256_broadcast_ss(pscale + row);
        for (i = 0; i + 7 < OC; i += 8) {
            auto vw = load_fp32x8(pw + i);
            // prefetch
            _mm_prefetch(pw + i + 1024, _MM_HINT_T1);
            auto vdst = _mm256_loadu_ps(dst + i);
            auto vsum = _mm256_fmadd_ps(vw, vscale, vdst);
            _mm256_storeu_ps(dst + i, vsum);
        }
    }

    static int32_t mem_mask[] = {0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1};
    auto remain_oc = (OC & 7);
    if (remain_oc) {
        auto vmask = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(&mem_mask[remain_oc]));
        i = OC - 8;
        for (g = 0; g < gate_cnt; g++) {
            auto row = gate_ids[g];
            auto pw = pw0 + row * stride_w;
            auto vscale = _mm256_broadcast_ss(pscale + row);
            auto vw = load_fp32x8(pw + i);
            auto vdst = _mm256_loadu_ps(dst + i);
            auto vsum = _mm256_fmadd_ps(vw, vscale, vdst);
            _mm256_maskstore_ps(dst + i, vmask, vsum);
        }
    }
}

/*
dst0 : [N, OC]
src0 : [num_copies, N, OC]
*/
static inline void reduce_outputs(float* dst0, float* src0, int num_copies, int N, int64_t OC) {
    static int32_t mem_mask[] = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};

    parallel_nt(0, [&](const int ithr, const int nthr) {
        int64_t oc0, oc1;
        splitter(OC, nthr, ithr, oc0, oc1);
        auto* dst = dst0;
        auto* src = src0;

        auto remain_oc = (oc1 - oc0) & 7;
        auto vmask = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(&mem_mask[8 - remain_oc]));
        for (int n = 0; n < N; n++, dst += OC, src += OC) {
            int i;
            for (i = oc0; i + 8 <= oc1; i += 8) {
                auto* ptemp = src + i;
                auto vsum = _mm256_setzero_ps();
                for (int k = 0; k < num_copies; k++, ptemp += N * OC) {
                    auto vw = _mm256_loadu_ps(ptemp);
                    vsum = _mm256_add_ps(vsum, vw);
                }
                _mm256_storeu_ps(dst + i, vsum);
            }
            if (i < oc1) {
                auto* ptemp = src + i;
                auto vsum = _mm256_setzero_ps();
                for (int k = 0; k < num_copies; k++, ptemp += N * OC) {
                    auto vw = _mm256_maskload_ps(ptemp, vmask);
                    vsum = _mm256_add_ps(vsum, vw);
                }
                _mm256_maskstore_ps(dst + i, vmask, vsum);
            }
            /*
            for (; i < oc1; i++) {
                auto* ptemp = src + i;
                auto vsum = 0.0f;
                for (int k = 0; k < num_copies; k++, ptemp += N*OC) {
                    vsum += ptemp[0];
                }
                dst[i] = vsum;
            }
            */
        }
    });
}

void sparse_accumulate_f32(float* dst,
                           int64_t output_channels,
                           float* weight_ptr,
                           int64_t weight_stride,
                           float* activation,
                           int64_t* sparse_ic_ptr,
                           int sparse_ic_cnt) {
    _sparse_accumulate<float>(dst,
                              output_channels,
                              weight_ptr,
                              weight_stride,
                              activation,
                              sparse_ic_ptr,
                              sparse_ic_cnt);
}

void sparse_accumulate_f16(float* dst,
                           int64_t output_channels,
                           ov::float16* weight_ptr,
                           int64_t weight_stride,
                           float* activation,
                           int64_t* sparse_ic_ptr,
                           int sparse_ic_cnt) {
    _sparse_accumulate<ov::float16>(dst,
                              output_channels,
                              weight_ptr,
                              weight_stride,
                              activation,
                              sparse_ic_ptr,
                              sparse_ic_cnt);
}


// x : [M, IC]
// W : [IC, OC]
//template <typename WType>
float dynPruneLinear(float* x, float* W2, float threshold, float zero_point, float* y2, int N, int IC, int OC) {
    // PROFILE(prof, "dynPruneLinear");
    // std::cout << "N=" << N << ", IC=" << IC << ", OC=" << OC << ", L=" << L << ", stride_x=" << stride_x << ",
    // stride_W=" << stride_W << std::endl;

    //==============
    // pruner
    // the sparse indices are ordered, and specially optimized for the case of N=1
    // sparse indices are index of input channel which contains big enough activations
    // in current batch
    static std::vector<int64_t> gate_ids;
    // PROFILE2(prof, "gate_ids");
    gate_ids.resize(IC);
    int64_t gate_cnt = 0;
    auto* base_x = x;
    auto stride_x = IC;
    for (int channel = 0; channel < IC; channel++) {
        // any non-zero gate value in current batch for channel oc
        // is detected
        auto* p_x = base_x + channel;
        for (int n = 0; n < N; n++, p_x += stride_x) {
            auto& value = p_x[n];
            if (std::abs(value - zero_point) > threshold) {
                gate_ids[gate_cnt] = channel;
                gate_cnt++;
                break;
            }
        }
    }

    // PROFILE2(prof, "accumulate2");
    auto stride_W2 = OC;
    auto stride_y2 = OC;

    auto nthr_max = parallel_get_max_threads();
    static std::vector<float> ytemp;
    ytemp.resize(nthr_max * N * OC);

    std::atomic_int actual_nthr;
    actual_nthr.store(0);

    parallel_nt(0, [&](const int ithr, const int nthr) {
        int64_t g0, g1;
        splitter(gate_cnt, nthr, ithr, g0, g1);

        actual_nthr++;
        auto* pdst = &ytemp[ithr * N * OC];
        memset(pdst, 0, N * OC * sizeof(float));
        auto* pids = gate_ids.data() + g0;
        auto* px = base_x;

        for (int n = 0; n < N; n++, pdst += OC, px += stride_x) {
            _sparse_accumulate(pdst,
                                OC,  // accumulation destination
                                W2,
                                stride_W2,  // W2 in row-major layout
                                px,
                                pids,
                                g1 - g0);  // gate_ids[i] select subrow of W2, multiply with py1[i] and accumulate
        }
    });

    // PROFILE2(prof, "reduce");
    reduce_outputs(y2, ytemp.data(), actual_nthr.load(), N, OC);
    return gate_cnt * 1.0f / IC;
}

#else
static inline void reduce_outputs(float* dst0, float* src0, int num_copies, int M, int64_t OC) {
    memset(dst0, 0, M * OC * sizeof(float));
    for (int i = 0; i < num_copies; i ++) {
        auto* src = &src0[i * M * OC];
        auto* dst = dst0;
        for (int m = 0; m < M; m++, src += OC, dst += OC) {
            for (int oc = 0; oc < OC; oc++)
                dst[oc] += src[oc];
        }
    }
}

float dynPruneLinear(float* x, float* W2, float threshold, float zero_point, float* y2, int N, int IC, int OC) {
    std::cout << "dynPruneLinear not supported\n";
    return 0;
}
#endif


static inline void accumulate_w8_peroc(int N,
                                    float* base_dst, int64_t OC,
                                    int* ic_ids, int ic_cnt,
                                    const uint8_t* Wu8,
                                    const uint8_t* zp,
                                    const float* scales,
                                    float* dense_x, int64_t IC) {
    // decompress zero-point
    thread_local std::vector<float> zpbuff;
    zpbuff.resize(OC);
    auto* dst_zp = zpbuff.data();

    int oc = 0;
#if defined(HAVE_AVX2)
    for (; oc + 8 <= OC; oc += 8) {
        auto zpu8 = _mm_loadu_si64(static_cast<void const*>(zp + oc));
        auto zpu32 = _mm256_cvtepu8_epi32(zpu8);
        auto zpf32 = _mm256_cvtepi32_ps(zpu32);
        _mm256_storeu_ps(dst_zp + oc, zpf32);
    }
#endif
    for (; oc < OC; oc ++) {
        dst_zp[oc] = zp[oc];
    }

    // vector x weights
    for (int g = 0; g < ic_cnt; g+=4) {
        auto ic0 = ic_ids[g];
        auto ic1 = ic_ids[g+1];
        auto ic2 = ic_ids[g+2];
        auto ic3 = ic_ids[g+3];

        const auto* p_w0 = Wu8 + ic0 * OC;
        const auto* p_w1 = Wu8 + ic1 * OC;
        const auto* p_w2 = Wu8 + ic2 * OC;
        const auto* p_w3 = Wu8 + ic3 * OC;

        oc = 0;
#if defined(HAVE_AVX2)
        auto vx0 = _mm256_broadcast_ss(dense_x + g + 0);
        auto vx1 = _mm256_broadcast_ss(dense_x + g + 1);
        auto vx2 = _mm256_broadcast_ss(dense_x + g + 2);
        auto vx3 = _mm256_broadcast_ss(dense_x + g + 3);
        for (; oc + 8 <= OC; oc += 8) {
            auto vscales = _mm256_loadu_ps(scales + oc);
            auto vzp = _mm256_loadu_ps(dst_zp + oc);

            auto vdst = _mm256_loadu_ps(base_dst + oc);
            auto wb0 = _mm_loadu_si64(static_cast<void const*>(p_w0 + oc));
            auto wb1 = _mm_loadu_si64(static_cast<void const*>(p_w1 + oc));
            auto wb2 = _mm_loadu_si64(static_cast<void const*>(p_w2 + oc));
            auto wb3 = _mm_loadu_si64(static_cast<void const*>(p_w3 + oc));
            auto vsum = _mm256_setzero_ps();

            auto wdw0 = _mm256_cvtepu8_epi32(wb0);
            vsum = _mm256_fmadd_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(wdw0), vzp), vx0, vsum);

            auto wdw1 = _mm256_cvtepu8_epi32(wb1);
            vsum = _mm256_fmadd_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(wdw1), vzp), vx1, vsum);

            auto wdw2 = _mm256_cvtepu8_epi32(wb2);
            vsum = _mm256_fmadd_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(wdw2), vzp), vx2, vsum);

            auto wdw3 = _mm256_cvtepu8_epi32(wb3);
            vsum = _mm256_fmadd_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(wdw3), vzp), vx3, vsum);

            vdst = _mm256_fmadd_ps(vsum, vscales, vdst);
            _mm256_storeu_ps(base_dst + oc, vdst);
        }
#endif
        if (oc < OC) {
            auto x0 = dense_x[g + 0];
            auto x1 = dense_x[g + 1];
            auto x2 = dense_x[g + 2];
            auto x3 = dense_x[g + 3];
            for (; oc < OC; oc ++) {
                auto weight0 = (p_w0[oc] - dst_zp[oc])*scales[oc];
                auto weight1 = (p_w1[oc] - dst_zp[oc])*scales[oc];
                auto weight2 = (p_w2[oc] - dst_zp[oc])*scales[oc];
                auto weight3 = (p_w3[oc] - dst_zp[oc])*scales[oc];
                base_dst[oc] += x0 * weight0 + x1 * weight1 + x2 * weight2 + x3 * weight3;
            }
        }
    }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gemm
#if defined(HAVE_AVX2)
template <int rows, int prefetch_v = 16>
void brgemm_6x2(const float* A,
                int A_stride,  // stride in number of element
                const float* B,
                int B_stride,  // stride in number of element
                float* C,
                int C_stride,  // stride in number of element
                int K,
                bool is_accumulate_C) {
    __m256 c0, c1, c2, c3, c4, c5;
    __m256 c6, c7, c8, c9, ca, cb;

    if (is_accumulate_C) {
        auto* src = C;
        c0 = _mm256_loadu_ps(src + 8 * 0);
        c1 = _mm256_loadu_ps(src + 8 * 1);
        if (rows > 1) {
            src += C_stride;
            c2 = _mm256_loadu_ps(src + 8 * 0);
            c3 = _mm256_loadu_ps(src + 8 * 1);
        }
        if (rows > 2) {
            src += C_stride;
            c4 = _mm256_loadu_ps(src + 8 * 0);
            c5 = _mm256_loadu_ps(src + 8 * 1);
        }
        if (rows > 3) {
            src += C_stride;
            c6 = _mm256_loadu_ps(src + 8 * 0);
            c7 = _mm256_loadu_ps(src + 8 * 1);
        }
        if (rows > 4) {
            src += C_stride;
            c8 = _mm256_loadu_ps(src + 8 * 0);
            c9 = _mm256_loadu_ps(src + 8 * 1);
        }
        if (rows > 5) {
            src += C_stride;
            ca = _mm256_loadu_ps(src + 8 * 0);
            cb = _mm256_loadu_ps(src + 8 * 1);
        }
    } else {
        c0 = _mm256_setzero_ps();
        c1 = _mm256_setzero_ps();
        c2 = _mm256_setzero_ps();
        c3 = _mm256_setzero_ps();
        c4 = _mm256_setzero_ps();
        c5 = _mm256_setzero_ps();
        c6 = _mm256_setzero_ps();
        c7 = _mm256_setzero_ps();
        c8 = _mm256_setzero_ps();
        c9 = _mm256_setzero_ps();
        ca = _mm256_setzero_ps();
        cb = _mm256_setzero_ps();
    }

    const auto* pA3 = A + 3 * A_stride;
    const auto prefetch_stride = B_stride * prefetch_v;
    int k;
    for (k = 0; k < K; k++, B += B_stride, A++, pA3++) {
        __m256 b0 = _mm256_loadu_ps(B + 8 * 0);
        __m256 b1 = _mm256_loadu_ps(B + 8 * 1);

        if (prefetch_v >= 0)
            _mm_prefetch(B + 16, _MM_HINT_T0);
        if (prefetch_v > 0)
            _mm_prefetch(B + prefetch_stride, _MM_HINT_T0);

        __m256 a0 = _mm256_broadcast_ss(A);
        c0 = _mm256_fmadd_ps(a0, b0, c0);
        c1 = _mm256_fmadd_ps(a0, b1, c1);
        if (rows > 1) {
            a0 = _mm256_broadcast_ss(A + A_stride);
            c2 = _mm256_fmadd_ps(a0, b0, c2);
            c3 = _mm256_fmadd_ps(a0, b1, c3);
        }
        if (rows > 2) {
            a0 = _mm256_broadcast_ss(A + 2 * A_stride);
            c4 = _mm256_fmadd_ps(a0, b0, c4);
            c5 = _mm256_fmadd_ps(a0, b1, c5);
        }

        if (rows > 3) {
            a0 = _mm256_broadcast_ss(pA3);
            c6 = _mm256_fmadd_ps(a0, b0, c6);
            c7 = _mm256_fmadd_ps(a0, b1, c7);
        }
        if (rows > 4) {
            a0 = _mm256_broadcast_ss(pA3 + A_stride);
            c8 = _mm256_fmadd_ps(a0, b0, c8);
            c9 = _mm256_fmadd_ps(a0, b1, c9);
        }
        if (rows > 5) {
            a0 = _mm256_broadcast_ss(pA3 + 2 * A_stride);
            ca = _mm256_fmadd_ps(a0, b0, ca);
            cb = _mm256_fmadd_ps(a0, b1, cb);
        }
    }

    // store C back
    _mm256_storeu_ps(C + 8 * 0, c0);
    _mm256_storeu_ps(C + 8 * 1, c1);
    if (rows > 1) {
        C += C_stride;
        _mm256_storeu_ps(C + 8 * 0, c2);
        _mm256_storeu_ps(C + 8 * 1, c3);
    }
    if (rows > 2) {
        C += C_stride;
        _mm256_storeu_ps(C + 8 * 0, c4);
        _mm256_storeu_ps(C + 8 * 1, c5);
    }
    if (rows > 3) {
        C += C_stride;
        _mm256_storeu_ps(C + 8 * 0, c6);
        _mm256_storeu_ps(C + 8 * 1, c7);
    }
    if (rows > 4) {
        C += C_stride;
        _mm256_storeu_ps(C + 8 * 0, c8);
        _mm256_storeu_ps(C + 8 * 1, c9);
    }
    if (rows > 5) {
        C += C_stride;
        _mm256_storeu_ps(C + 8 * 0, ca);
        _mm256_storeu_ps(C + 8 * 1, cb);
    }
}

template <class T>
static T* scratch_alloc(size_t cnt) {
    thread_local uint8_t scratch[1024 * 1024 * 2] __attribute__((aligned(4096)));
    // assert(cnt * sizeof(T) < sizeof(scratch));
    // DEBUG_LOG(reinterpret_cast<void*>(scratch));
    return reinterpret_cast<T*>(scratch);
}

void MM_ComputeBounded_reuseA_i8(
            const float * A,
            float * C,
            const uint8_t* W,
            const uint8_t* zp,
            const float* scales,
            int M, int IC, int OC,
            int n0, int n1) {
    constexpr int BK = 54;
    float* scratch = scratch_alloc<float>(16 * BK + OC);

    int K = IC;
    auto A_stride = IC;
    auto C_stride = OC;
    auto W_stride = OC;

    void (*brgmm6x2_tail)(const float*, int, const float*, int, float*, int, int, bool) = nullptr;
    auto M_tails = M % 6;
    auto M_body = M - M_tails;
    switch (M_tails) {
    case 5:
        brgmm6x2_tail = &brgemm_6x2<5>;
        break;
    case 4:
        brgmm6x2_tail = &brgemm_6x2<4>;
        break;
    case 3:
        brgmm6x2_tail = &brgemm_6x2<3>;
        break;
    case 2:
        brgmm6x2_tail = &brgemm_6x2<2>;
        break;
    case 1:
        brgmm6x2_tail = &brgemm_6x2<1>;
        break;
    }

    float* repacked_B = scratch;
    float* zero_points = scratch + 16 * BK;

    // deocompress zero-point into scratch
    int n = 0;
    for (n = n0; n + 8 <= n1; n += 8) {
        auto vzpu8 = _mm_loadu_si64(static_cast<void const*>(zp + n));
        auto vzpf32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(vzpu8));
        _mm256_storeu_ps(zero_points + n - n0, vzpf32);
    }
    for (; n < n1; n ++) {
        zero_points[n - n0] = zp[n];
    }

    for (int k = 0; k < K; k += BK, A += BK, W += BK * W_stride) {
        int bK = std::min(K - k, BK);
        auto is_accumulate_C = (k > 0);

        for (int n = n0; n + 2 * 8 <= n1; n += 2 * 8) {
            // prepack [BK, 16] into scratch
            {
                const auto* srcW = W + n;
                auto* dst = repacked_B;

                auto vzp0 = _mm256_loadu_ps(zero_points + (n - n0));
                auto vzp1 = _mm256_loadu_ps(zero_points + (n - n0) + 8);
                auto vscale0 = _mm256_loadu_ps(scales + n);
                auto vscale1 = _mm256_loadu_ps(scales + n + 8);

                for (int k = 0; k < bK; k++, dst += 2 * 8, srcW += W_stride) {
                    auto wb0 = _mm_loadu_si64(static_cast<void const*>(srcW + 8 * 0));
                    auto wb1 = _mm_loadu_si64(static_cast<void const*>(srcW + 8 * 1));
                    auto wf0 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(wb0)), vzp0);
                    auto wf1 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(wb1)), vzp1);
                    wf0 = _mm256_mul_ps(wf0, vscale0);
                    wf1 = _mm256_mul_ps(wf1, vscale1);

                    // prefetch right
                    _mm_prefetch(srcW + 64, _MM_HINT_T0);

                    _mm256_storeu_ps(dst + 8 * 0, wf0);
                    _mm256_storeu_ps(dst + 8 * 1, wf1);
                }
            }

            auto *pA = A;
            auto* pC = C + n;
            for (int m = 0; m < M_body; m += 6, pA += 6 * A_stride, pC += 6 * C_stride) {
                brgemm_6x2<6>(pA, A_stride, repacked_B, 2 * 8, pC, C_stride, bK, is_accumulate_C);
            }
            if (brgmm6x2_tail)
                (*brgmm6x2_tail)(pA, A_stride, repacked_B, 2 * 8, pC, C_stride, bK, is_accumulate_C);
        }
    }
}
#else
void MM_ComputeBounded_reuseA_i8(
            const float * A,
            float * C,
            const uint8_t* W,
            const uint8_t* zp,
            const float* scales,
            int M, int IC, int OC,
            int n0, int n1) {
}
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void dynPruneLinear_i8(const float* input,      // [M, IC]
                        float threshold,
                        float zero_point,
                        const uint8_t* W,       // [IC, OC]
                        const uint8_t* zp,      // [OC]
                        const float* scales,    // [OC]
                        float* output,          // [M, OC]
                        int M, int IC, int OC) {
    if ((OC % 8) > 0) {
        throw std::runtime_error("OC is not multiple of 8");
    }

    auto prof = LinuxPerf::Profile("gate_ids");
    if (M > 1) {
        parallel_nt(0, [&](const int ithr, const int nthr) {
            int n0, n1;
            splitter(OC/8, nthr, ithr, n0, n1);
            n0 *= 8;
            n1 *= 8;
            MM_ComputeBounded_reuseA_i8(
                input, output,
                W, zp, scales, M, IC, OC, n0, n1);
        });
        return;
        // M > 1 is much less likely to benefit from sparsity
        for (int m = 0; m < M; m++, input += IC, output += OC) {
            for (int oc = 0; oc < OC; oc++)
                output[oc] = 0;

            for (int ic = 0; ic < IC; ic++) {
                float x = input[ic];
                if (std::abs(x - zero_point) < threshold) continue;

                auto * pw = W + ic*OC;
                int oc = 0;
#if defined(HAVE_AVX2)
                auto vx = _mm256_broadcast_ss(input + ic);
                for (; oc + 8 <= OC; oc += 8) {
                    auto vw = _mm_loadu_si64(static_cast<void const*>(pw + oc));
                    auto vzp = _mm_loadu_si64(static_cast<void const*>(zp + oc));
                    auto vscale = _mm256_loadu_ps(scales + oc);

                    auto vdst = _mm256_loadu_ps(output + oc);
                    auto wdecomp = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(vw)),
                                                  _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(vzp)));
                    wdecomp = _mm256_mul_ps(wdecomp, vscale);
                    vdst = _mm256_fmadd_ps(vx, wdecomp, vdst);
                    _mm256_storeu_ps(output + oc, vdst);
                }
#endif
                for (; oc < OC; oc ++) {
                    auto weight = static_cast<float>(pw[oc]) - static_cast<float>(zp[oc]);
                    output[oc] += x * weight * scales[oc];
                }
            }
        }
        return;
    }

    static std::vector<int> gate_ids;
    static std::vector<float> gate_val;
    int gate_cnt = 0;
    gate_ids.resize(IC);
    gate_val.resize(IC);
    for (int channel = 0; channel < IC; channel++) {
        auto* src = input + channel;
        for (int m = 0; m < M; m++, src += IC) {
            auto& value = src[m];
            if (std::abs(value - zero_point) >= threshold) {
                gate_ids[gate_cnt] = channel;
                gate_val[gate_cnt] = value;
                gate_cnt++;
                break;
            }
        }
    }

    // pad to 4
    auto last_channel = gate_ids[gate_cnt - 1];
    while (gate_cnt & 3) {
        gate_ids[gate_cnt] = last_channel;
        gate_val[gate_cnt] = 0.0f;
        gate_cnt++;
    }

    // std::cout << M << "," << IC << "," << OC << "," << threshold << "," << zero_point << std::endl;
    prof = LinuxPerf::Profile("mm");

    // this mm kernel is the most time-consuming one
    auto nthr_max = parallel_get_max_threads();
    static std::vector<float> output_temp;
    output_temp.resize(nthr_max * M * OC);

    parallel_nt(0, [&](const int ithr, const int nthr) {
        int g0, g1;
        splitter(gate_cnt/4, nthr, ithr, g0, g1);
        g0 *= 4;
        g1 *= 4;

        auto* pdst = &output_temp[ithr * M * OC];
        memset(pdst, 0, M * OC * sizeof(output_temp[0]));

        if (M == 1) {
            accumulate_w8_peroc(1, pdst, OC, &gate_ids[g0], g1 - g0, W, zp, scales, &gate_val[g0], IC);
        } else {
            thread_local std::vector<float> weight_deq;
            weight_deq.resize(OC);

            for (int g = g0; g < g1; g++) {
                int ic = gate_ids[g];

                // dequantize-weight for ic
                auto* pw = W + ic * OC;
                for (int oc = 0; oc < OC; oc++) {
                    weight_deq[oc] = (static_cast<float>(pw[oc]) - static_cast<float>(zp[oc])) * scales[oc];
                }

                // accumulate dequantized weight into output vectors
                auto* src = input;
                auto* dst = &output_temp[ithr * M * OC];
                for (int m = 0; m < M; m++, src += IC, dst += OC) {
                    auto x = src[ic];
                    for (int oc = 0; oc < OC; oc++) {
                        dst[oc] += x * weight_deq[oc];
                    }
                }
            }
        }
    });

    prof = LinuxPerf::Profile("reduce");
    reduce_outputs(output, output_temp.data(), nthr_max, M, OC);
    return;
}

/******************************************************************
for AWQ's INT4 group-128 compressed weight matrix, the basic procedure is the same as above except for the second step, 
in which we tried harder to lower the computational cost required for decompressing the row vectors of the weight matrix
from INT4 into float32 (so that we can reach the memory bound instead of compute bound):

* the zero-points of AWQ's weight matrix is also compressed in INT4 and requiring decompression into higher
  precision before being subtracted from compressed weight, and it's one per-1-output-channel and
  per-128-input-channels-group, thus we decompress a row of zero-point of the group only once into
  a small scrach buffer and reuse it when iterating over input channels belonging to same group.

* iterate weight matrix in unit of row requires frequently load & store of the scratch accumulation buffer,
  in our case this buffer is usually small and well cached (w/o incurring memory bandwidth penalty), but
  these instructions still consume computational resources, thus we iterate 4 neighbouring salient channels
  in parallel to amortize this cost.

* decompression INT4 into float32 is commonly done using vpmovsxbd & vcvtdq2ps instructions which has high
  latency and low throughput, here we directly embed 4bit unsigned integer into the mantissa part of the
  IEEE-754 encoded single precision float value 1.0, this can be done efficiently using vpsrld/vpslld,vandps
  and vorps, and we will get (1.0 + w/16) in float32 format, by decompressing zero-points in same way we have
  (1.0 + zp/16), after subtraction we can cancel out the constant 1.0 and get (w-zp)/16, and by multiplying
  input by 16 (which is done only once before the loop) we can cancel out the 1/16 factor after FMA w/o any
  additional cost. this allows us to lower the computational cost and reach the memory bound better.

for M > 1 case, we choose to ignore sparsity (based on the assumption that accuracy won't be impacted)
and implement a simple brgemm-based kernel, since the weight layout is choosen to prefer M=1 case, we
cannot change weight-layout for M>1 case.

note: for INT8 weight compressed in per-OC only (no per-group)
******************************************************************/

void dynPruneLinear_i8_opt(const float* input,
                        float threshold,
                        float zero_point,
                        const uint8_t* W,
                        const uint8_t* zp,
                        const float* scales,
                        float* output, int M, int IC, int OC) {
    static std::vector<int> gate_ids;
    int gate_cnt = 0;
    gate_ids.resize(IC);
    for (int channel = 0; channel < IC; channel++) {
        auto* src = input + channel;
        for (int m = 0; m < M; m++, src += IC) {
            auto& value = src[m];
            if (std::abs(value - zero_point) > threshold) {
                gate_ids[gate_cnt] = channel;
                gate_cnt++;
                break;
            }
        }
    }

    auto nthr_max = parallel_get_max_threads();
    static std::vector<float> output_temp;
    output_temp.resize(nthr_max * M * OC);

    parallel_nt(0, [&](const int ithr, const int nthr) {
        int g0, g1;
        splitter(gate_cnt, nthr, ithr, g0, g1);

        thread_local std::vector<float> weight_deq;
        weight_deq.resize(OC);

        memset(&output_temp[ithr * M * OC], 0, M * OC * sizeof(output_temp[0]));

        for (int g = g0; g < g1; g++) {
            int ic = gate_ids[g];

            // dequantize-weight for ic
            auto* pw = W + ic * OC;
            for (int oc = 0; oc < OC; oc++) {
                weight_deq[oc] = (static_cast<float>(pw[oc]) - static_cast<float>(zp[oc])) * scales[oc];
            }

            // accumulate dequantized weight into output vectors
            auto* src = input;
            auto* dst = &output_temp[ithr * M * OC];
            for (int m = 0; m < M; m++, src += IC, dst += OC) {
                auto x = src[ic];
                for (int oc = 0; oc < OC; oc++) {
                    dst[oc] += x * weight_deq[oc];
                }
            }
        }
    });

    reduce_outputs(output, output_temp.data(), nthr_max, M, OC);
    return;
}


}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov