// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cstring>
#include "act_sparse_fc_kernel.hpp"

#include "openvino/core/parallel.hpp"

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
    return 0;
}
#endif

void dynPruneLinear_i8(const float* input,
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