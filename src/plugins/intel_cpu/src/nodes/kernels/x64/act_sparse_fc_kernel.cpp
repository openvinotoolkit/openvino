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


static inline void accumulate_w4_peroc(float* base_dst, int OC,
                                       int* ic_ids, int ic_cnt,
                                       const uint8_t* W,
                                       const uint8_t* zp,
                                       const float* scales,
                                       float* dense_x, int IC, int IC_group_size) {
    // decompress zero-point
    thread_local std::vector<float> zpbuff;
    zpbuff.resize(OC);
    int last_gid = -1;
    // vector x weights
    for (int g = 0; g < ic_cnt; g+=4) {
        auto ic0 = ic_ids[g];
        auto ic1 = ic_ids[g+1];
        auto ic2 = ic_ids[g+2];
        auto ic3 = ic_ids[g+3];
        auto gid = ic0 / IC_group_size;
        auto* p_scales = scales + gid*OC;

        // entering a new group, decompress zero-points
        if (last_gid != gid) {
            auto* dst_zp = zpbuff.data();
            auto* src_zp = zp + gid * (OC/2);
            int oc = 0;
        #if defined(HAVE_AVX2)
            auto vmask_u4 = _mm256_set1_epi32(0xF);
            for (; oc + 16 <= OC; oc += 16, src_zp += 8) {
                auto vzp16xu4 = _mm_loadu_si64(static_cast<void const*>(src_zp));
                auto vzp16xu4_i32 = _mm256_cvtepu8_epi32(vzp16xu4);
                auto vzp16xu4_i32_low = _mm256_and_si256(vzp16xu4_i32, vmask_u4);
                auto vzp16xu4_i32_high = _mm256_srli_epi32(vzp16xu4_i32, 4);
                auto vzpf32_low = _mm256_cvtepi32_ps(vzp16xu4_i32_low);
                auto vzpf32_high = _mm256_cvtepi32_ps(vzp16xu4_i32_high);
                _mm256_storeu_ps(dst_zp + oc, vzpf32_low);
                _mm256_storeu_ps(dst_zp + oc + 8, vzpf32_high);
            }
        #endif
            for (; oc < OC; oc +=2, src_zp++) {
                dst_zp[oc] = src_zp[0] & 0xF;
                dst_zp[oc + 1] = src_zp[0] >> 4;
            }
            last_gid = gid;
        }

        const auto* p_w0 = W + ic0 * OC/2;
        const auto* p_w1 = W + ic1 * OC/2;
        const auto* p_w2 = W + ic2 * OC/2;
        const auto* p_w3 = W + ic3 * OC/2;
        auto* dst_zp = zpbuff.data();

        int oc = 0;
#if defined(HAVE_AVX2)
        auto vmask_u4 = _mm256_set1_epi32(0xF);
        auto vx0 = _mm256_broadcast_ss(dense_x + g + 0);
        auto vx1 = _mm256_broadcast_ss(dense_x + g + 1);
        auto vx2 = _mm256_broadcast_ss(dense_x + g + 2);
        auto vx3 = _mm256_broadcast_ss(dense_x + g + 3);
        for (; oc + 16 <= OC; oc += 16) {
            auto vzp0 = _mm256_loadu_ps(dst_zp + oc);
            auto vzp1 = _mm256_loadu_ps(dst_zp + oc + 8);

            auto vdst0 = _mm256_loadu_ps(base_dst + oc);
            auto vdst1 = _mm256_loadu_ps(base_dst + oc + 8);

            auto wb0 = _mm_loadu_si64(static_cast<void const*>(p_w0)); p_w0 += 8;
            auto vsum0 = _mm256_setzero_ps();
            auto vsum1 = _mm256_setzero_ps();

            auto wdw_i32 = _mm256_cvtepu8_epi32(wb0);
            auto wdw0 = _mm256_cvtepi32_ps(_mm256_and_si256(wdw_i32, vmask_u4));
            auto wdw1 = _mm256_cvtepi32_ps(_mm256_srli_epi32(wdw_i32, 4));
            vsum0 = _mm256_fmadd_ps(_mm256_sub_ps(wdw0, vzp0), vx0, vsum0);
            vsum1 = _mm256_fmadd_ps(_mm256_sub_ps(wdw1, vzp1), vx0, vsum1);

            wb0 = _mm_loadu_si64(static_cast<void const*>(p_w1)); p_w1 += 8;
            wdw_i32 = _mm256_cvtepu8_epi32(wb0);
            wdw0 = _mm256_cvtepi32_ps(_mm256_and_si256(wdw_i32, vmask_u4));
            wdw1 = _mm256_cvtepi32_ps(_mm256_srli_epi32(wdw_i32, 4));
            vsum0 = _mm256_fmadd_ps(_mm256_sub_ps(wdw0, vzp0), vx1, vsum0);
            vsum1 = _mm256_fmadd_ps(_mm256_sub_ps(wdw1, vzp1), vx1, vsum1);

            wb0 = _mm_loadu_si64(static_cast<void const*>(p_w2)); p_w2 += 8;
            wdw_i32 = _mm256_cvtepu8_epi32(wb0);
            wdw0 = _mm256_cvtepi32_ps(_mm256_and_si256(wdw_i32, vmask_u4));
            wdw1 = _mm256_cvtepi32_ps(_mm256_srli_epi32(wdw_i32, 4));
            vsum0 = _mm256_fmadd_ps(_mm256_sub_ps(wdw0, vzp0), vx2, vsum0);
            vsum1 = _mm256_fmadd_ps(_mm256_sub_ps(wdw1, vzp1), vx2, vsum1);

            wb0 = _mm_loadu_si64(static_cast<void const*>(p_w3)); p_w3 += 8;
            wdw_i32 = _mm256_cvtepu8_epi32(wb0);
            wdw0 = _mm256_cvtepi32_ps(_mm256_and_si256(wdw_i32, vmask_u4));
            wdw1 = _mm256_cvtepi32_ps(_mm256_srli_epi32(wdw_i32, 4));
            vsum0 = _mm256_fmadd_ps(_mm256_sub_ps(wdw0, vzp0), vx3, vsum0);
            vsum1 = _mm256_fmadd_ps(_mm256_sub_ps(wdw1, vzp1), vx3, vsum1);

            auto vscales0 = _mm256_loadu_ps(p_scales + oc);
            auto vscales1 = _mm256_loadu_ps(p_scales + oc + 8);

            vdst0 = _mm256_fmadd_ps(vsum0, vscales0, vdst0);
            vdst1 = _mm256_fmadd_ps(vsum1, vscales1, vdst1);
            _mm256_storeu_ps(base_dst + oc, vdst0);
            _mm256_storeu_ps(base_dst + oc + 8, vdst1);
        }
#endif
        if (oc < OC) {
            auto x0 = dense_x[g + 0];
            auto x1 = dense_x[g + 1];
            auto x2 = dense_x[g + 2];
            auto x3 = dense_x[g + 3];
            for (; oc < OC; oc += 16, p_w0 += 8, p_w1 += 8, p_w2 += 8, p_w3 += 8) {
                for (int i = 0; i < 8; i++) {
                    auto zero_point = dst_zp[oc + i];
                    auto scale = p_scales[oc + i];
                    auto weight0 = (p_w0[i] & 0xF) - zero_point;
                    auto weight1 = (p_w1[i] & 0xF) - zero_point;
                    auto weight2 = (p_w2[i] & 0xF) - zero_point;
                    auto weight3 = (p_w3[i] & 0xF) - zero_point;
                    base_dst[oc + i] += (x0 * weight0 + x1 * weight1 + x2 * weight2 + x3 * weight3) * scale;
                }
                for (int i = 0; i < 8; i++) {
                    auto zero_point = dst_zp[oc + i + 8];
                    auto scale = p_scales[oc + i + 8];
                    auto weight0 = (p_w0[i] >> 4) - zero_point;
                    auto weight1 = (p_w1[i] >> 4) - zero_point;
                    auto weight2 = (p_w2[i] >> 4) - zero_point;
                    auto weight3 = (p_w3[i] >> 4) - zero_point;
                    base_dst[oc + i + 8] += (x0 * weight0 + x1 * weight1 + x2 * weight2 + x3 * weight3) * scale;
                }
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
                auto* dst = repacked_B;
                auto vzp0 = _mm256_loadu_ps(zero_points + (n - n0));
                auto vzp1 = _mm256_loadu_ps(zero_points + (n - n0) + 8);
                auto vscale0 = _mm256_loadu_ps(scales + n);
                auto vscale1 = _mm256_loadu_ps(scales + n + 8);
                const auto* srcW = W + n;
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

void MM_ComputeBounded_reuseA_i4(
            const float * A,
            float * C,
            const uint8_t* W,
            const uint8_t* zp,
            const float* scales,
            int M, int IC, int OC,
            int n0, int n1, int icgs) {
    int BK = icgs;
    float* scratch = scratch_alloc<float>(16 * BK + OC);

    int K = IC;
    auto A_stride = IC;
    auto C_stride = OC;
    auto W_stride = (OC/2);

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

    for (int k = 0; k < K; k += BK, A += BK, W += BK * W_stride, zp += W_stride, scales += OC) {
        int bK = std::min(K - k, BK);
        auto is_accumulate_C = (k > 0);

        // deocompress zero-point into scratch buffer
        {
            const auto* pzp = zp;
            int n = 0;
            auto vmask_u4 = _mm256_set1_epi32(0xF);
            for (n = n0; n + 16 <= n1; n += 16, pzp += 8) {
                auto vzp16xu4 = _mm_loadu_si64(static_cast<void const*>(pzp));
                // 8 x low-4bits  : 0,1,2,3,4,5,6,7
                // 8 x high-4bits : 8,9,a,b,c,d,e,f
                auto vzp16xu4_i32 = _mm256_cvtepu8_epi32(vzp16xu4);

                auto vzp16xu4_i32_low = _mm256_and_si256(vzp16xu4_i32, vmask_u4);
                auto vzp16xu4_i32_high = _mm256_srli_epi32(vzp16xu4_i32, 4);

                auto vzpf32_low = _mm256_cvtepi32_ps(vzp16xu4_i32_low);
                auto vzpf32_high = _mm256_cvtepi32_ps(vzp16xu4_i32_high);
                _mm256_storeu_ps(zero_points + n - n0, vzpf32_low);
                _mm256_storeu_ps(zero_points + n - n0 + 8, vzpf32_high);
            }
            for (; n < n1; n += 2, pzp++) {
                zero_points[n - n0] = (*pzp) & 0xF;
                zero_points[n - n0 + 1] = (*pzp) >> 4;
            }
        }

        for (int n = n0; n + 2 * 8 <= n1; n += 2 * 8) {
            // prepack subB [BK, 16] into scratch
            // because BK is fully contained within IC-group (BK == n*IC_group_size), it can share same zp & scales
            {
                auto* dst = repacked_B;
                auto vzp0 = _mm256_loadu_ps(zero_points + (n - n0));
                auto vzp1 = _mm256_loadu_ps(zero_points + (n - n0) + 8);
                auto vscale0 = _mm256_loadu_ps(scales + n);
                auto vscale1 = _mm256_loadu_ps(scales + n + 8);
                const auto* srcW = W + n/2;
                auto vmask_u4 = _mm256_set1_epi32(0xF);
                for (int k = 0; k < bK; k++, dst += 2 * 8, srcW += W_stride) {
                    // 16 x i4
                    auto wb0 = _mm_loadu_si64(static_cast<void const*>(srcW + 8 * 0));
                    auto wdw = _mm256_cvtepu8_epi32(wb0);

                    auto wdw_low = _mm256_and_si256(wdw, vmask_u4);
                    auto wdw_high = _mm256_srli_epi32(wdw, 4);

                    auto wf0 = _mm256_sub_ps(_mm256_cvtepi32_ps(wdw_low), vzp0);
                    auto wf1 = _mm256_sub_ps(_mm256_cvtepi32_ps(wdw_high), vzp1);
                    wf0 = _mm256_mul_ps(wf0, vscale0);
                    wf1 = _mm256_mul_ps(wf1, vscale1);

                    // prefetch right
                    _mm_prefetch(srcW + 64, _MM_HINT_T0);

                    _mm256_storeu_ps(dst + 8 * 0, wf0);
                    _mm256_storeu_ps(dst + 8 * 1, wf1);
                }
            }

            // matmul(A, subB)
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
void MM_ComputeBounded_reuseA_i4(
            const float * A,
            float * C,
            const uint8_t* W,
            const uint8_t* zp,
            const float* scales,
            int M, int IC, int OC,
            int n0, int n1, int icgs) {
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
    }

    auto prof = LinuxPerf::Profile("gate_ids");
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
        accumulate_w8_peroc(1, pdst, OC, &gate_ids[g0], g1 - g0, W, zp, scales, &gate_val[g0], IC);
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
void dynPruneLinear_i4(const float* input,      // [M, IC]
                        float threshold,
                        float zero_point,
                        const uint8_t* W,       // [IC, OC]
                        const uint8_t* zp,      // [OC]
                        const float* scales,    // [OC]
                        float* output,          // [M, OC]
                        int M, int IC, int OC,
                        int IC_group_size) {
    if ((OC % 8) > 0) {
        throw std::runtime_error("OC is not multiple of 8");
    }

    if (M > 1) {
        // a reference impl
        parallel_nt(0, [&](const int ithr, const int nthr) {
            int n0, n1;
            splitter(OC/8, nthr, ithr, n0, n1);
            n0 *= 8;
            n1 *= 8;
            MM_ComputeBounded_reuseA_i4(
                input, output,
                W, zp, scales, M, IC, OC, n0, n1, IC_group_size);
        });
        return;
    }

    auto prof = LinuxPerf::Profile("gate_ids");
    static std::vector<int> gate_ids;
    static std::vector<float> gate_val;
    int gate_cnt = 0;
    gate_ids.resize(IC);
    gate_val.resize(IC);
    for (int c0 = 0; c0 < IC; c0 += IC_group_size) {
        for (int c1 = 0; c1 < IC_group_size; c1++) {
            auto channel = c0 + c1;
            auto& value = input[channel];
            if (std::abs(value - zero_point) >= threshold) {
                gate_ids[gate_cnt] = channel;
                gate_val[gate_cnt] = value;
                gate_cnt++;
            }
        }
        if (gate_cnt & 3) {
            // padding : ensuer 4 rows are from same group
            auto n_pad = 4 - (gate_cnt & 3);
            auto ic_pad = gate_ids[gate_cnt-1];
            for (int i = 0; i < n_pad; i++) {
                gate_ids[gate_cnt] = ic_pad;
                gate_val[gate_cnt] = 0.0f;
                gate_cnt++;
            }
        }
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
        accumulate_w4_peroc(pdst, OC, &gate_ids[g0], g1 - g0, W, zp, scales, &gate_val[g0], IC, IC_group_size);
    });

    prof = LinuxPerf::Profile("reduce");
    reduce_outputs(output, output_temp.data(), nthr_max, M, OC);
    return;
}

// [OC, IC/2, 2] => [IC, OC/2, 2]
// each row is further reordered in unit of 16 x i4 in [0,8,1,9,2,a,3,b,4,c,5,d,6,e,7,f] order
void dynPruneLinear_repack_i4(uint8_t * src, uint8_t * dst, int IC, int OC) {
    auto src_stride = IC / 2;

    int ic = 0;
#if defined(HAVE_AVX2)
    uint8_t scratch0[64];
    uint8_t scratch1[64];
    for (ic = 0; ic + 2*32 <= IC; ic += 2*32) {
        // 64-ic
        auto* pdst = dst + ic * (OC / 2);
        auto vmask_low_u4 = _mm256_set1_epi8(0xF);
        auto vmask_high_u4 = _mm256_set1_epi8(0xF0);
        for (int oc = 0; oc < OC; oc += 16, pdst += 8) {
            // 64-ic x 16-oc
            auto* psrc_oc0 = src + (ic / 2) + (oc + 0)*src_stride;
            auto* psrc_oc8 = src + (ic / 2) + (oc + 8)*src_stride;
            for (int k = 0; k < 8; k++, psrc_oc0 += src_stride, psrc_oc8 += src_stride) {
                // oc+0: ic0~64
                auto b0 = _mm256_loadu_si256(reinterpret_cast<__m256i const *>(psrc_oc0));
                // oc+8: ic0~64
                auto b8 = _mm256_loadu_si256(reinterpret_cast<__m256i const *>(psrc_oc8));
                auto b0_ic0 = _mm256_and_si256(b0, vmask_low_u4);
                auto b0_ic1 = _mm256_and_si256(_mm256_srli_epi16(b0, 4), vmask_low_u4);

                auto b8_ic0 = _mm256_and_si256(_mm256_slli_epi16(b8, 4), vmask_high_u4);
                auto b8_ic1 = _mm256_and_si256(b8, vmask_high_u4);

                auto bdst_ic0 = _mm256_or_si256(b8_ic0, b0_ic0);    // even channels
                auto bdst_ic1 = _mm256_or_si256(b8_ic1, b0_ic1);    // odd channels

                _mm256_storeu_si256(reinterpret_cast<__m256i *>(scratch0), bdst_ic0);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(scratch1), bdst_ic1);

                auto* pdst_temp0 = pdst + k;
                auto* pdst_temp1 = pdst + k + (OC / 2);
                for (int i = 0; i < 32; i++, pdst_temp0 += OC, pdst_temp1 += OC) {
                    *pdst_temp0 = scratch0[i];
                    *pdst_temp1 = scratch1[i];
                }
            }
        }
    }
#endif
    // tails
    for (; ic < IC; ic += 2) {
        auto* pdst_a = dst + ic * (OC / 2);
        auto* pdst_b = pdst_a + (OC / 2);
        for (int oc = 0; oc < OC; oc += 16, pdst_a += 8, pdst_b += 8) {
            auto* psrc_oc0 = src + (ic / 2) + (oc + 0)*src_stride;
            auto* psrc_oc8 = src + (ic / 2) + (oc + 8)*src_stride;
            for (int k = 0; k < 8; k++, psrc_oc0 += src_stride, psrc_oc8 += src_stride) {
                auto data0 = *psrc_oc0;  // [ic1, ic0] packed in same u8
                auto u40a = (data0 & 0xF);
                auto u40b = (data0 >> 4);
                auto data8 = *psrc_oc8;
                auto u48a = (data8 & 0xF);
                auto u48b = (data8 >> 4);
                pdst_a[k] = (u48a << 4) | u40a;
                pdst_b[k] = (u48b << 4) | u40b;
            }
        }
    }
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov