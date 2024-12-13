// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cstring>
#include "act_sparse_fc_kernel.hpp"

#include "openvino/core/parallel.hpp"

//#include "/home/tingqian/aboutSHW/include/linux_perf.hpp"
//#include "/home/openvino-ci-58/tingqian/aboutSHW/include/linux_perf.hpp"

#define PROFILE(x) LinuxPerf::Profile(x)
#define PROFILE(x) 1

#include "simd.hpp"

// https://github.com/intel-sandbox/dynSparseFC/blob/main/dyn_sparse_fc.cpp

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {
// gemm
template <int rows, int prefetch_v = 16>
void brgemm_6x2(const float* A,
                int A_stride,  // stride in number of element
                const float* B,
                int B_stride,  // stride in number of element
                float* C,
                int C_stride,  // stride in number of element
                int K,
                bool is_accumulate_C) {
    SIMD_F32 c0, c1, c2, c3, c4, c5;
    SIMD_F32 c6, c7, c8, c9, ca, cb;

    if (is_accumulate_C) {
        auto* src = C;
        c0 = simd_loadu_ps(src + SIMDW * 0);
        c1 = simd_loadu_ps(src + SIMDW * 1);
        if (rows > 1) {
            src += C_stride;
            c2 = simd_loadu_ps(src + SIMDW * 0);
            c3 = simd_loadu_ps(src + SIMDW * 1);
        }
        if (rows > 2) {
            src += C_stride;
            c4 = simd_loadu_ps(src + SIMDW * 0);
            c5 = simd_loadu_ps(src + SIMDW * 1);
        }
        if (rows > 3) {
            src += C_stride;
            c6 = simd_loadu_ps(src + SIMDW * 0);
            c7 = simd_loadu_ps(src + SIMDW * 1);
        }
        if (rows > 4) {
            src += C_stride;
            c8 = simd_loadu_ps(src + SIMDW * 0);
            c9 = simd_loadu_ps(src + SIMDW * 1);
        }
        if (rows > 5) {
            src += C_stride;
            ca = simd_loadu_ps(src + SIMDW * 0);
            cb = simd_loadu_ps(src + SIMDW * 1);
        }
    } else {
        c0 = simd_setzero_ps();
        c1 = simd_setzero_ps();
        c2 = simd_setzero_ps();
        c3 = simd_setzero_ps();
        c4 = simd_setzero_ps();
        c5 = simd_setzero_ps();
        c6 = simd_setzero_ps();
        c7 = simd_setzero_ps();
        c8 = simd_setzero_ps();
        c9 = simd_setzero_ps();
        ca = simd_setzero_ps();
        cb = simd_setzero_ps();
    }

    const auto* pA3 = A + 3 * A_stride;
    const auto prefetch_stride = B_stride * prefetch_v;
    int k;
    for (k = 0; k < K; k++, B += B_stride, A++, pA3++) {
        auto b0 = simd_loadu_ps(B + SIMDW * 0);
        auto b1 = simd_loadu_ps(B + SIMDW * 1);

        if (prefetch_v >= 0)
            simd_prefetch(B + 16, _MM_HINT_T0);
        if (prefetch_v > 0)
            simd_prefetch(B + prefetch_stride, _MM_HINT_T0);

        auto a0 = simd_broadcast_ss(A);
        c0 = simd_fmadd_ps(a0, b0, c0);
        c1 = simd_fmadd_ps(a0, b1, c1);
        if (rows > 1) {
            a0 = simd_broadcast_ss(A + A_stride);
            c2 = simd_fmadd_ps(a0, b0, c2);
            c3 = simd_fmadd_ps(a0, b1, c3);
        }
        if (rows > 2) {
            a0 = simd_broadcast_ss(A + 2 * A_stride);
            c4 = simd_fmadd_ps(a0, b0, c4);
            c5 = simd_fmadd_ps(a0, b1, c5);
        }

        if (rows > 3) {
            a0 = simd_broadcast_ss(pA3);
            c6 = simd_fmadd_ps(a0, b0, c6);
            c7 = simd_fmadd_ps(a0, b1, c7);
        }
        if (rows > 4) {
            a0 = simd_broadcast_ss(pA3 + A_stride);
            c8 = simd_fmadd_ps(a0, b0, c8);
            c9 = simd_fmadd_ps(a0, b1, c9);
        }
        if (rows > 5) {
            a0 = simd_broadcast_ss(pA3 + 2 * A_stride);
            ca = simd_fmadd_ps(a0, b0, ca);
            cb = simd_fmadd_ps(a0, b1, cb);
        }
    }

    // store C back
    simd_storeu_ps(C + SIMDW * 0, c0);
    simd_storeu_ps(C + SIMDW * 1, c1);
    if (rows > 1) {
        C += C_stride;
        simd_storeu_ps(C + SIMDW * 0, c2);
        simd_storeu_ps(C + SIMDW * 1, c3);
    }
    if (rows > 2) {
        C += C_stride;
        simd_storeu_ps(C + SIMDW * 0, c4);
        simd_storeu_ps(C + SIMDW * 1, c5);
    }
    if (rows > 3) {
        C += C_stride;
        simd_storeu_ps(C + SIMDW * 0, c6);
        simd_storeu_ps(C + SIMDW * 1, c7);
    }
    if (rows > 4) {
        C += C_stride;
        simd_storeu_ps(C + SIMDW * 0, c8);
        simd_storeu_ps(C + SIMDW * 1, c9);
    }
    if (rows > 5) {
        C += C_stride;
        simd_storeu_ps(C + SIMDW * 0, ca);
        simd_storeu_ps(C + SIMDW * 1, cb);
    }
}

template <int row>
void brgemm_4x3(const float* A,
                int A_stride,  // stride in number of element
                const float* B,
                int B_stride,  // stride in number of element
                float* C,
                int C_stride,  // stride in number of element
                int K,
                bool is_accumulate_C) {
    // loop in unit of register blocking: (3x4*8)
    SIMD_F32 c00, c01, c02;
    SIMD_F32 c10, c11, c12;
    SIMD_F32 c20, c21, c22;
    SIMD_F32 c30, c31, c32;

    if (is_accumulate_C) {
        c00 = simd_loadu_ps(C + SIMDW * 0);
        c01 = simd_loadu_ps(C + SIMDW * 1);
        c02 = simd_loadu_ps(C + SIMDW * 2);

        if (row > 1) {
            c10 = simd_loadu_ps(C + C_stride + SIMDW * 0);
            c11 = simd_loadu_ps(C + C_stride + SIMDW * 1);
            c12 = simd_loadu_ps(C + C_stride + SIMDW * 2);
        }
        if (row > 2) {
            c20 = simd_loadu_ps(C + 2 * C_stride + SIMDW * 0);
            c21 = simd_loadu_ps(C + 2 * C_stride + SIMDW * 1);
            c22 = simd_loadu_ps(C + 2 * C_stride + SIMDW * 2);
        }
        if (row > 3) {
            c30 = simd_loadu_ps(C + 3 * C_stride + SIMDW * 0);
            c31 = simd_loadu_ps(C + 3 * C_stride + SIMDW * 1);
            c32 = simd_loadu_ps(C + 3 * C_stride + SIMDW * 2);
        }
    } else {
        c00 = simd_setzero_ps();
        c01 = simd_setzero_ps();
        c02 = simd_setzero_ps();

        if (row > 1) {
            c10 = simd_setzero_ps();
            c11 = simd_setzero_ps();
            c12 = simd_setzero_ps();
        }
        if (row > 2) {
            c20 = simd_setzero_ps();
            c21 = simd_setzero_ps();
            c22 = simd_setzero_ps();
        }
        if (row > 3) {
            c30 = simd_setzero_ps();
            c31 = simd_setzero_ps();
            c32 = simd_setzero_ps();
        }
    }

    auto* prefetch_B = B + 64 / sizeof(float) * 10;

    // reducing along k dimension
    //   with -O2 optimization flag, following kernel has 6~7 cycles-per-iteration
    //   which is consistent with FMA's throughput(0.5)
    for (int k = 0; k < K; k++, B += B_stride, A++, prefetch_B += B_stride) {
        // 16-ymm-registers are just enough for 4x3 register blocking
        auto b0 = simd_loadu_ps(B + SIMDW * 0);
        auto b1 = simd_loadu_ps(B + SIMDW * 1);
        auto b2 = simd_loadu_ps(B + SIMDW * 2);

        //_mm_prefetch(prefetch_B, _MM_HINT_T0);

        auto a = simd_broadcast_ss(A);
        c00 = simd_fmadd_ps(a, b0, c00);
        c01 = simd_fmadd_ps(a, b1, c01);
        c02 = simd_fmadd_ps(a, b2, c02);

        if (row > 1) {
            a = simd_broadcast_ss(A + A_stride);
            c10 = simd_fmadd_ps(a, b0, c10);
            c11 = simd_fmadd_ps(a, b1, c11);
            c12 = simd_fmadd_ps(a, b2, c12);
        }
        if (row > 2) {
            a = simd_broadcast_ss(A + 2 * A_stride);
            c20 = simd_fmadd_ps(a, b0, c20);
            c21 = simd_fmadd_ps(a, b1, c21);
            c22 = simd_fmadd_ps(a, b2, c22);
        }
        if (row > 3) {
            a = simd_broadcast_ss(A + 3 * A_stride);
            c30 = simd_fmadd_ps(a, b0, c30);
            c31 = simd_fmadd_ps(a, b1, c31);
            c32 = simd_fmadd_ps(a, b2, c32);
        }
    }

    // store C back
    simd_storeu_ps(C + SIMDW * 0, c00);
    simd_storeu_ps(C + SIMDW * 1, c01);
    simd_storeu_ps(C + SIMDW * 2, c02);
    if (row > 1) {
        simd_storeu_ps(C + C_stride + SIMDW * 0, c10);
        simd_storeu_ps(C + C_stride + SIMDW * 1, c11);
        simd_storeu_ps(C + C_stride + SIMDW * 2, c12);
    }
    if (row > 2) {
        simd_storeu_ps(C + 2 * C_stride + SIMDW * 0, c20);
        simd_storeu_ps(C + 2 * C_stride + SIMDW * 1, c21);
        simd_storeu_ps(C + 2 * C_stride + SIMDW * 2, c22);
    }
    if (row > 3) {
        simd_storeu_ps(C + 3 * C_stride + SIMDW * 0, c30);
        simd_storeu_ps(C + 3 * C_stride + SIMDW * 1, c31);
        simd_storeu_ps(C + 3 * C_stride + SIMDW * 2, c32);
    }
}

template <int row>
void brgemm_4x1(const float* A,
                int A_stride,  // stride in number of element
                const float* B,
                int B_stride,  // stride in number of element
                float* C,
                int C_stride,  // stride in number of element
                int K,
                bool is_accumulate_C) {
    // loop in unit of register blocking: (3x4*8)
    SIMD_F32 c00;
    SIMD_F32 c10;
    SIMD_F32 c20;
    SIMD_F32 c30;

    if (is_accumulate_C) {
        c00 = simd_loadu_ps(C + 8 * 0);
        if (row > 1) {
            c10 = simd_loadu_ps(C + C_stride + SIMDW * 0);
        }
        if (row > 2) {
            c20 = simd_loadu_ps(C + 2 * C_stride + SIMDW * 0);
        }
        if (row > 3) {
            c30 = simd_loadu_ps(C + 3 * C_stride + SIMDW * 0);
        }
    } else {
        c00 = simd_setzero_ps();
        if (row > 1) {
            c10 = simd_setzero_ps();
        }
        if (row > 2) {
            c20 = simd_setzero_ps();
        }
        if (row > 3) {
            c30 = simd_setzero_ps();
        }
    }

    // reducing along k dimension
    //   with -O2 optimization flag, following kernel has 6~7 cycles-per-iteration
    //   which is consistent with FMA's throughput(0.5)
    for (int k = 0; k < K; k++, B += B_stride, A++) {
        // 16-ymm-registers are just enough for 4x3 register blocking
        auto b0 = simd_loadu_ps(B + 8 * 0);
        auto a = simd_broadcast_ss(A);
        c00 = simd_fmadd_ps(a, b0, c00);
        if (row > 1) {
            a = simd_broadcast_ss(A + A_stride);
            c10 = simd_fmadd_ps(a, b0, c10);
        }
        if (row > 2) {
            a = simd_broadcast_ss(A + 2 * A_stride);
            c20 = simd_fmadd_ps(a, b0, c20);
        }
        if (row > 3) {
            a = simd_broadcast_ss(A + 3 * A_stride);
            c30 = simd_fmadd_ps(a, b0, c30);
        }
    }
    simd_storeu_ps(C + SIMDW * 0, c00);
    if (row > 1) {
        simd_storeu_ps(C + C_stride + SIMDW * 0, c10);
    }
    if (row > 2) {
        simd_storeu_ps(C + 2 * C_stride + SIMDW * 0, c20);
    }
    if (row > 3) {
        simd_storeu_ps(C + 3 * C_stride + SIMDW * 0, c30);
    }
}

/*
dst               : result
pw0/stride_w      : weight matrix in [IC, OC] layout
pscale            : input activations
gate_ids/gate_cnt : an array of input channels with non-zero activations (after thresholding)
*/
void accumulate_wf16(float* dst,
                        int64_t OC,
                        int* gate_ids,
                        int gate_cnt,
                        const ov::float16* pw0,
                        float* dense_x,
                        int IC) {
    int i = 0;
    int g = 0;
    int64_t stride_w = OC;
    for (; g < gate_cnt; g += 4) {
        auto row0 = gate_ids[g];
        auto row1 = gate_ids[g + 1];
        auto row2 = gate_ids[g + 2];
        auto row3 = gate_ids[g + 3];
        auto p_w0 = pw0 + row0 * stride_w;
        auto p_w1 = pw0 + row1 * stride_w;
        auto p_w2 = pw0 + row2 * stride_w;
        auto p_w3 = pw0 + row3 * stride_w;
        auto vscale0 = simd_broadcast_ss(dense_x + g);
        auto vscale1 = simd_broadcast_ss(dense_x + g + 1);
        auto vscale2 = simd_broadcast_ss(dense_x + g + 2);
        auto vscale3 = simd_broadcast_ss(dense_x + g + 3);
        for (i = 0; i + SIMDW <= OC; i += SIMDW) {
            auto vdst = simd_loadu_ps(dst + i);
            auto vw0 = simd_loadu_ps(p_w0 + i);
            auto vw1 = simd_loadu_ps(p_w1 + i);
            auto vw2 = simd_loadu_ps(p_w2 + i);
            auto vw3 = simd_loadu_ps(p_w3 + i);
            // prefetch
            //_mm_prefetch(p_w0 + i + 1024, _MM_HINT_T1);
            //_mm_prefetch(p_w1 + i + 1024, _MM_HINT_T1);
            //_mm_prefetch(p_w2 + i + 1024, _MM_HINT_T1);
            //_mm_prefetch(p_w3 + i + 1024, _MM_HINT_T1);
            vdst = simd_fmadd_ps(vw0, vscale0, vdst);
            vdst = simd_fmadd_ps(vw1, vscale1, vdst);
            vdst = simd_fmadd_ps(vw2, vscale2, vdst);
            vdst = simd_fmadd_ps(vw3, vscale3, vdst);
            simd_storeu_ps(dst + i, vdst);
        }
    }

    auto remain_oc = (OC & 7);
    if (remain_oc) {
        simd_mask mask(remain_oc);
        i = OC - 8;
        for (g = 0; g < gate_cnt; g++) {
            auto row = gate_ids[g];
            auto pw = pw0 + row * stride_w;
            auto vscale = simd_broadcast_ss(dense_x + g);
            auto vw = mask.load(pw + i);
            auto vdst = mask.load(dst + i);
            auto vsum = simd_fmadd_ps(vw, vscale, vdst);
            mask.store(dst + i, vsum);
        }
    }
}

/*
dst0 : [N, OC]
src0 : [num_copies, N, OC]
*/
static inline void reduce_outputs(float* dst0, float* src0, int num_copies, int N, int64_t OC) {
    parallel_nt(0, [&](const int ithr, const int nthr) {
        int64_t oc0, oc1;
        splitter(OC/SIMDW, nthr, ithr, oc0, oc1);
        oc0 *= SIMDW;
        oc1 *= SIMDW;
        if (oc1 > OC) oc1 = OC;

        auto* dst = dst0;
        auto* src = src0;

        auto remain_oc = (oc1 - oc0) % SIMDW;
        simd_mask mask(remain_oc);
        for (int n = 0; n < N; n++, dst += OC, src += OC) {
            int i;
            for (i = oc0; i + SIMDW <= oc1; i += SIMDW) {
                auto* ptemp = src + i;
                auto vsum = simd_setzero_ps();
                for (int k = 0; k < num_copies; k++, ptemp += N * OC) {
                    auto vw = simd_loadu_ps(ptemp);
                    vsum = simd_add_ps(vsum, vw);
                }
                simd_storeu_ps(dst + i, vsum);
            }
            if (i < oc1) {
                auto* ptemp = src + i;
                auto vsum = simd_setzero_ps();
                for (int k = 0; k < num_copies; k++, ptemp += N * OC) {
                    auto vw = mask.load(ptemp);
                    vsum = simd_add_ps(vsum, vw);
                }
                mask.store(dst + i, vsum);
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

template <class T>
static T* scratch_alloc(size_t cnt) {
    thread_local uint8_t scratch[1024 * 1024 * 2] __attribute__((aligned(4096)));
    // assert(cnt * sizeof(T) < sizeof(scratch));
    // DEBUG_LOG(reinterpret_cast<void*>(scratch));
    return reinterpret_cast<T*>(scratch);
}

void MM_ComputeBounded_reuseA_f16(const float* A,
                                  float* C,
                                  const ov::float16* W,
                                  int M,
                                  int IC,
                                  int OC,
                                  int n0,
                                  int n1) {
    constexpr int BK = 54;
    float* scratch = scratch_alloc<float>(BK * (SIMDW*2) + OC);

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

    for (int k = 0; k < K; k += BK, A += BK, W += BK * W_stride) {
        int bK = std::min(K - k, BK);
        auto is_accumulate_C = (k > 0);

        for (int n = n0; n + 2 * SIMDW <= n1; n += 2 * SIMDW) {
            // prepack [BK, 16] into scratch
            auto* dst = repacked_B;
            const auto* srcW = W + n;
            for (int k = 0; k < bK; k++, dst += 2 * SIMDW, srcW += W_stride) {
                auto wf0 = simd_loadu_ps(srcW + SIMDW * 0);
                auto wf1 = simd_loadu_ps(srcW + SIMDW * 1);
                // prefetch right
                simd_prefetch(srcW + 64, _MM_HINT_T0);
                simd_storeu_ps(dst + SIMDW * 0, wf0);
                simd_storeu_ps(dst + SIMDW * 1, wf1);
            }

            auto *pA = A;
            auto* pC = C + n;
            for (int m = 0; m < M_body; m += 6, pA += 6 * A_stride, pC += 6 * C_stride) {
                brgemm_6x2<6>(pA, A_stride, repacked_B, 2 * SIMDW, pC, C_stride, bK, is_accumulate_C);
            }
            if (brgmm6x2_tail)
                (*brgmm6x2_tail)(pA, A_stride, repacked_B, 2 * SIMDW, pC, C_stride, bK, is_accumulate_C);
        }
    }
}

// x : [M, IC]
// W : [IC, OC]
//template <typename WType>
void dynPruneLinear_f16(const float* input,
                       float threshold,
                       float zero_point,
                       const ov::float16* W,
                       float* output,
                       int M,
                       int IC,
                       int OC) {
    if (M > 1) {
        parallel_nt(0, [&](const int ithr, const int nthr) {
            int n0, n1;
            splitter(OC / (2 * SIMDW), nthr, ithr, n0, n1);
            n0 *= 2 * SIMDW;
            n1 *= 2 * SIMDW;
            MM_ComputeBounded_reuseA_f16(input, output, W, M, IC, OC, n0, n1);
        });
        return;
    }

    auto prof = PROFILE("gate_ids");
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
        accumulate_wf16(pdst, OC, &gate_ids[g0], g1 - g0, W, &gate_val[g0], IC);
    });

    prof = PROFILE("reduce");
    reduce_outputs(output, output_temp.data(), nthr_max, M, OC);
}

template<bool with_zp>
void accumulate_w8_peroc(float* base_dst, int64_t OC,
                        int* ic_ids, int ic_cnt,
                        const uint8_t* Wu8,
                        const uint8_t* zp,
                        const float* scales,
                        float* dense_x, int64_t IC) {
    // decompress zero-point
    thread_local std::vector<float> zpbuff;
    zpbuff.resize(OC);
    auto* dst_zp = zpbuff.data();

    if (with_zp) {
        int oc = 0;
        for (; oc + SIMDW <= OC; oc += SIMDW) {
            auto zpu32 = simd_load_epu8_epi32(static_cast<void const*>(zp + oc));
            auto zpf32 = simd_cvtepi32_ps(zpu32);
            simd_storeu_ps(dst_zp + oc, zpf32);
        }
        for (; oc < OC; oc ++) {
            dst_zp[oc] = zp[oc];
        }
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

        int oc = 0;

        auto vx0 = simd_broadcast_ss(dense_x + g + 0);
        auto vx1 = simd_broadcast_ss(dense_x + g + 1);
        auto vx2 = simd_broadcast_ss(dense_x + g + 2);
        auto vx3 = simd_broadcast_ss(dense_x + g + 3);
        for (; oc + SIMDW <= OC; oc += SIMDW) {
            if (with_zp) {
                auto vscales = simd_loadu_ps(scales + oc);
                auto vzp = simd_loadu_ps(dst_zp + oc);
                auto vdst = simd_loadu_ps(base_dst + oc);

                auto wdw0 = simd_load_epu8_epi32(static_cast<void const*>(p_w0 + oc));
                auto wdw1 = simd_load_epu8_epi32(static_cast<void const*>(p_w1 + oc));
                auto wdw2 = simd_load_epu8_epi32(static_cast<void const*>(p_w2 + oc));
                auto wdw3 = simd_load_epu8_epi32(static_cast<void const*>(p_w3 + oc));

                auto vsum = simd_setzero_ps();

                vsum = simd_fmadd_ps(simd_sub_ps(simd_cvtepi32_ps(wdw0), vzp), vx0, vsum);
                vsum = simd_fmadd_ps(simd_sub_ps(simd_cvtepi32_ps(wdw1), vzp), vx1, vsum);
                vsum = simd_fmadd_ps(simd_sub_ps(simd_cvtepi32_ps(wdw2), vzp), vx2, vsum);
                vsum = simd_fmadd_ps(simd_sub_ps(simd_cvtepi32_ps(wdw3), vzp), vx3, vsum);

                vdst = simd_fmadd_ps(vsum, vscales, vdst);
                simd_storeu_ps(base_dst + oc, vdst);
            } else {
                auto vscales = simd_loadu_ps(scales + oc);
                auto vdst = simd_loadu_ps(base_dst + oc);

                auto wdw0 = simd_load_epi8_epi32(static_cast<void const*>(p_w0 + oc));
                auto wdw1 = simd_load_epi8_epi32(static_cast<void const*>(p_w1 + oc));
                auto wdw2 = simd_load_epi8_epi32(static_cast<void const*>(p_w2 + oc));
                auto wdw3 = simd_load_epi8_epi32(static_cast<void const*>(p_w3 + oc));

                auto vsum = simd_setzero_ps();

                vsum = simd_fmadd_ps(simd_cvtepi32_ps(wdw0), vx0, vsum);
                vsum = simd_fmadd_ps(simd_cvtepi32_ps(wdw1), vx1, vsum);
                vsum = simd_fmadd_ps(simd_cvtepi32_ps(wdw2), vx2, vsum);
                vsum = simd_fmadd_ps(simd_cvtepi32_ps(wdw3), vx3, vsum);

                vdst = simd_fmadd_ps(vsum, vscales, vdst);
                simd_storeu_ps(base_dst + oc, vdst);
            }
        }

        if (oc < OC) {
            auto x0 = dense_x[g + 0];
            auto x1 = dense_x[g + 1];
            auto x2 = dense_x[g + 2];
            auto x3 = dense_x[g + 3];
            for (; oc < OC; oc ++) {
                auto weight0 = p_w0[oc];
                auto weight1 = p_w1[oc];
                auto weight2 = p_w2[oc];
                auto weight3 = p_w3[oc];
                if (with_zp) {
                    weight0 -= dst_zp[oc];
                    weight1 -= dst_zp[oc];
                    weight2 -= dst_zp[oc];
                    weight3 -= dst_zp[oc];
                }
                weight0 *= scales[oc];
                weight1 *= scales[oc];
                weight2 *= scales[oc];
                weight3 *= scales[oc];
                base_dst[oc] += x0 * weight0 + x1 * weight1 + x2 * weight2 + x3 * weight3;
            }
        }
    }
}

template<bool with_zp>
void accumulate_w4(float* base_dst, int OC,
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
            if (with_zp) {
                auto* dst_zp = zpbuff.data();
                auto* src_zp = zp + gid * (OC/2);
                int oc = 0;
                auto vmask_u4 = simd_set1_epi32(0xF);
                for (; oc + SIMDW*2 <= OC; oc += SIMDW*2, src_zp += SIMDW) {
                    auto vzp16xu4_i32 = simd_load_epu8_epi32(static_cast<void const*>(src_zp));
                    auto vzp16xu4_i32_low = simd_and_si(vzp16xu4_i32, vmask_u4);
                    auto vzp16xu4_i32_high = simd_srli_epi32(vzp16xu4_i32, 4);
                    auto vzpf32_low = simd_cvtepi32_ps(vzp16xu4_i32_low);
                    auto vzpf32_high = simd_cvtepi32_ps(vzp16xu4_i32_high);
                    simd_storeu_ps(dst_zp + oc, vzpf32_low);
                    simd_storeu_ps(dst_zp + oc + SIMDW, vzpf32_high);
                }
                for (; oc < OC; oc +=2, src_zp++) {
                    dst_zp[oc] = src_zp[0] & 0xF;
                    dst_zp[oc + 1] = src_zp[0] >> 4;
                }
            }
            last_gid = gid;
        }

        const auto* p_w0 = W + ic0 * OC/2;
        const auto* p_w1 = W + ic1 * OC/2;
        const auto* p_w2 = W + ic2 * OC/2;
        const auto* p_w3 = W + ic3 * OC/2;
        auto* dst_zp = zpbuff.data();

        int oc = 0;

        auto vmask_u4 = simd_set1_epi32(0xF);
        auto vx0 = simd_broadcast_ss(dense_x + g + 0);
        auto vx1 = simd_broadcast_ss(dense_x + g + 1);
        auto vx2 = simd_broadcast_ss(dense_x + g + 2);
        auto vx3 = simd_broadcast_ss(dense_x + g + 3);
        if (with_zp) {
            for (; oc + SIMDW*2 <= OC; oc += SIMDW*2) {
                auto vzp0 = simd_loadu_ps(dst_zp + oc);
                auto vzp1 = simd_loadu_ps(dst_zp + oc + SIMDW);

                auto vdst0 = simd_loadu_ps(base_dst + oc);
                auto vdst1 = simd_loadu_ps(base_dst + oc + SIMDW);

                auto wdw_i32 = simd_load_epu8_epi32(static_cast<void const*>(p_w0)); p_w0 += SIMDW;
                auto vsum0 = simd_setzero_ps();
                auto vsum1 = simd_setzero_ps();

                auto wdw0 = simd_cvtepi32_ps(simd_and_si(wdw_i32, vmask_u4));
                auto wdw1 = simd_cvtepi32_ps(simd_srli_epi32(wdw_i32, 4));
                vsum0 = simd_fmadd_ps(simd_sub_ps(wdw0, vzp0), vx0, vsum0);
                vsum1 = simd_fmadd_ps(simd_sub_ps(wdw1, vzp1), vx0, vsum1);

                wdw_i32 = simd_load_epu8_epi32(static_cast<void const*>(p_w1)); p_w1 += SIMDW;
                wdw0 = simd_cvtepi32_ps(simd_and_si(wdw_i32, vmask_u4));
                wdw1 = simd_cvtepi32_ps(simd_srli_epi32(wdw_i32, 4));
                vsum0 = simd_fmadd_ps(simd_sub_ps(wdw0, vzp0), vx1, vsum0);
                vsum1 = simd_fmadd_ps(simd_sub_ps(wdw1, vzp1), vx1, vsum1);

                wdw_i32 = simd_load_epu8_epi32(static_cast<void const*>(p_w2)); p_w2 += SIMDW;
                wdw0 = simd_cvtepi32_ps(simd_and_si(wdw_i32, vmask_u4));
                wdw1 = simd_cvtepi32_ps(simd_srli_epi32(wdw_i32, 4));
                vsum0 = simd_fmadd_ps(simd_sub_ps(wdw0, vzp0), vx2, vsum0);
                vsum1 = simd_fmadd_ps(simd_sub_ps(wdw1, vzp1), vx2, vsum1);

                wdw_i32 = simd_load_epu8_epi32(static_cast<void const*>(p_w3)); p_w3 += SIMDW;
                wdw0 = simd_cvtepi32_ps(simd_and_si(wdw_i32, vmask_u4));
                wdw1 = simd_cvtepi32_ps(simd_srli_epi32(wdw_i32, 4));
                vsum0 = simd_fmadd_ps(simd_sub_ps(wdw0, vzp0), vx3, vsum0);
                vsum1 = simd_fmadd_ps(simd_sub_ps(wdw1, vzp1), vx3, vsum1);

                auto vscales0 = simd_loadu_ps(p_scales + oc);
                auto vscales1 = simd_loadu_ps(p_scales + oc + SIMDW);

                vdst0 = simd_fmadd_ps(vsum0, vscales0, vdst0);
                vdst1 = simd_fmadd_ps(vsum1, vscales1, vdst1);
                simd_storeu_ps(base_dst + oc, vdst0);
                simd_storeu_ps(base_dst + oc + SIMDW, vdst1);
            }
        } else {
            for (; oc + SIMDW*2 <= OC; oc += SIMDW*2) {
                auto vdst0 = simd_loadu_ps(base_dst + oc);
                auto vdst1 = simd_loadu_ps(base_dst + oc + SIMDW);

                auto wdw_i32 = simd_load_epu8_epi32(static_cast<void const*>(p_w0)); p_w0 += SIMDW;
                auto vsum0 = simd_setzero_ps();
                auto vsum1 = simd_setzero_ps();

                auto wdw0 = simd_cvtepi32_ps(simd_srai_epi32(simd_slli_epi32(wdw_i32, 32-4), 32-4));
                auto wdw1 = simd_cvtepi32_ps(simd_srai_epi32(simd_slli_epi32(wdw_i32, 32-8), 32-4));
                vsum0 = simd_fmadd_ps(wdw0, vx0, vsum0);
                vsum1 = simd_fmadd_ps(wdw1, vx0, vsum1);

                wdw_i32 = simd_load_epu8_epi32(static_cast<void const*>(p_w1)); p_w1 += SIMDW;

                wdw0 = simd_cvtepi32_ps(simd_srai_epi32(simd_slli_epi32(wdw_i32, 32-4), 32-4));
                wdw1 = simd_cvtepi32_ps(simd_srai_epi32(simd_slli_epi32(wdw_i32, 32-8), 32-4));
                vsum0 = simd_fmadd_ps(wdw0, vx1, vsum0);
                vsum1 = simd_fmadd_ps(wdw1, vx1, vsum1);

                wdw_i32 = simd_load_epu8_epi32(static_cast<void const*>(p_w2)); p_w2 += SIMDW;
                wdw0 = simd_cvtepi32_ps(simd_srai_epi32(simd_slli_epi32(wdw_i32, 32-4), 32-4));
                wdw1 = simd_cvtepi32_ps(simd_srai_epi32(simd_slli_epi32(wdw_i32, 32-8), 32-4));
                vsum0 = simd_fmadd_ps(wdw0, vx2, vsum0);
                vsum1 = simd_fmadd_ps(wdw1, vx2, vsum1);

                wdw_i32 = simd_load_epu8_epi32(static_cast<void const*>(p_w3)); p_w3 += SIMDW;
                wdw0 = simd_cvtepi32_ps(simd_srai_epi32(simd_slli_epi32(wdw_i32, 32-4), 32-4));
                wdw1 = simd_cvtepi32_ps(simd_srai_epi32(simd_slli_epi32(wdw_i32, 32-8), 32-4));
                vsum0 = simd_fmadd_ps(wdw0, vx3, vsum0);
                vsum1 = simd_fmadd_ps(wdw1, vx3, vsum1);

                auto vscales0 = simd_loadu_ps(p_scales + oc);
                auto vscales1 = simd_loadu_ps(p_scales + oc + SIMDW);

                vdst0 = simd_fmadd_ps(vsum0, vscales0, vdst0);
                vdst1 = simd_fmadd_ps(vsum1, vscales1, vdst1);
                simd_storeu_ps(base_dst + oc, vdst0);
                simd_storeu_ps(base_dst + oc + SIMDW, vdst1);
            }
        }

        if (oc < OC) {
            auto x0 = dense_x[g + 0];
            auto x1 = dense_x[g + 1];
            auto x2 = dense_x[g + 2];
            auto x3 = dense_x[g + 3];
            for (; oc < OC; oc += SIMDW*2, p_w0 += SIMDW, p_w1 += SIMDW, p_w2 += SIMDW, p_w3 += SIMDW) {
                for (int i = 0; i < SIMDW; i++) {
                    auto scale = p_scales[oc + i];
                    float weight0;
                    float weight1;
                    float weight2;
                    float weight3;
                    if (with_zp) {
                        auto zero_point = dst_zp[oc + i];
                        weight0 = (p_w0[i] & 0xF) - zero_point;
                        weight1 = (p_w1[i] & 0xF) - zero_point;
                        weight2 = (p_w2[i] & 0xF) - zero_point;
                        weight3 = (p_w3[i] & 0xF) - zero_point;
                    } else {
                        weight0 = (reinterpret_cast<const int8_t*>(p_w0)[i] << 4) >> 4;
                        weight1 = (reinterpret_cast<const int8_t*>(p_w1)[i] << 4) >> 4;
                        weight2 = (reinterpret_cast<const int8_t*>(p_w2)[i] << 4) >> 4;
                        weight3 = (reinterpret_cast<const int8_t*>(p_w3)[i] << 4) >> 4;
                    }
                    base_dst[oc + i] += (x0 * weight0 + x1 * weight1 + x2 * weight2 + x3 * weight3) * scale;
                }
                for (int i = 0; i < SIMDW; i++) {
                    auto scale = p_scales[oc + i + SIMDW];
                    float weight0;
                    float weight1;
                    float weight2;
                    float weight3;
                    if (with_zp) {
                        auto zero_point = dst_zp[oc + i + SIMDW];
                        weight0 = (p_w0[i] >> 4) - zero_point;
                        weight1 = (p_w1[i] >> 4) - zero_point;
                        weight2 = (p_w2[i] >> 4) - zero_point;
                        weight3 = (p_w3[i] >> 4) - zero_point;
                    } else {
                        weight0 = (reinterpret_cast<const int8_t*>(p_w0)[i] >> 4);
                        weight1 = (reinterpret_cast<const int8_t*>(p_w1)[i] >> 4);
                        weight2 = (reinterpret_cast<const int8_t*>(p_w2)[i] >> 4);
                        weight3 = (reinterpret_cast<const int8_t*>(p_w3)[i] >> 4);
                    }
                    base_dst[oc + i + SIMDW] += (x0 * weight0 + x1 * weight1 + x2 * weight2 + x3 * weight3) * scale;
                }
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<bool with_zp>
void repack_weight_for_4x3(const uint8_t* W, int strideW, const float* scales, const float* zp, int K, int N, float* repacked_B_nx3, float* repacked_B_nx1) {
    //assert((N % 8) == 0);
#if 1
    for (int k = 0; k < K; k++) {
        int n0 = 0;
        auto* src = W + k*strideW;
        auto* dst = repacked_B_nx3 + k*SIMDW*3;
        auto dst_stride = K*SIMDW*3;
        for (n0 = 0; n0 + SIMDW*3 <= N; n0 += SIMDW*3, dst += dst_stride) {
            SIMD_F32 wf0;
            SIMD_F32 wf1;
            SIMD_F32 wf2;
            if (with_zp) {
                auto wi0 = simd_load_epu8_epi32(static_cast<void const*>(src + n0 + SIMDW * 0));
                auto wi1 = simd_load_epu8_epi32(static_cast<void const*>(src + n0 + SIMDW * 1));
                auto wi2 = simd_load_epu8_epi32(static_cast<void const*>(src + n0 + SIMDW * 2));
                auto zp0 = simd_loadu_ps(zp + n0 + SIMDW * 0);
                auto zp1 = simd_loadu_ps(zp + n0 + SIMDW * 1);
                auto zp2 = simd_loadu_ps(zp + n0 + SIMDW * 2);
                wf0 = simd_sub_ps(simd_cvtepi32_ps(wi0), (zp0));
                wf1 = simd_sub_ps(simd_cvtepi32_ps(wi1), (zp1));
                wf2 = simd_sub_ps(simd_cvtepi32_ps(wi2), (zp2));
            } else {
                auto wi0 = simd_load_epi8_epi32(static_cast<void const*>(src + n0 + SIMDW * 0));
                auto wi1 = simd_load_epi8_epi32(static_cast<void const*>(src + n0 + SIMDW * 1));
                auto wi2 = simd_load_epi8_epi32(static_cast<void const*>(src + n0 + SIMDW * 2));
                wf0 = simd_cvtepi32_ps(wi0);
                wf1 = simd_cvtepi32_ps(wi1);
                wf2 = simd_cvtepi32_ps(wi2);
            }
            wf0 = simd_mul_ps(wf0, simd_loadu_ps(scales + n0 + SIMDW*0));
            wf1 = simd_mul_ps(wf1, simd_loadu_ps(scales + n0 + SIMDW*1));
            wf2 = simd_mul_ps(wf2, simd_loadu_ps(scales + n0 + SIMDW*2));
            simd_storeu_ps(dst + SIMDW*0, wf0);
            simd_storeu_ps(dst + SIMDW*1, wf1);
            simd_storeu_ps(dst + SIMDW*2, wf2);
        }

        dst = repacked_B_nx1 + k*SIMDW;
        dst_stride = K*SIMDW;
        for (; n0 < N; n0 += SIMDW, dst += dst_stride) {
            for (int n = n0; n < n0+SIMDW; n++) {
                SIMD_F32 wf0;
                if (with_zp) {
                    auto wi0 = simd_load_epu8_epi32(static_cast<void const*>(src + n0 + SIMDW * 0));
                    auto zp0 = simd_loadu_ps(zp + n0 + SIMDW * 0);
                    wf0 = simd_sub_ps(simd_cvtepi32_ps(wi0), (zp0));
                } else {
                    auto wi0 = simd_load_epi8_epi32(static_cast<void const*>(src + n0 + SIMDW * 0));
                    wf0 = simd_cvtepi32_ps(wi0);
                }
                wf0 = simd_mul_ps(wf0, simd_loadu_ps(scales + n0 + SIMDW*0));
                simd_storeu_ps(dst + SIMDW*0, wf0);
            }
        }
    }
#else
    for (int k = 0; k < K; k++) {
        int n0 = 0;
        auto* src = W + k*strideW;
        auto* dst = repacked_B_nx3 + k*SIMDW*3;
        auto dst_stride = K*SIMDW*3;
        for (n0 = 0; n0 + SIMDW*3 <= N; n0 += SIMDW*3, dst += dst_stride) {
            for (int n = n0; n < n0+SIMDW*3; n++) {
                dst[n-n0] = (src[n] - zp[n]) * scales[n];
                //printf("%d,%d,%d  %d, %f, %f, =>  %f\n", k, n0, n, src[n], zp[n], scales[n], dst[n-n0]);
            }
        }
        dst = repacked_B_nx1 + k*SIMDW;
        dst_stride = K*SIMDW;
        for (; n0 < N; n0 += SIMDW, dst += dst_stride) {
            for (int n = n0; n < n0+SIMDW; n++) {
                dst[n-n0] = (src[n] - zp[n]) * scales[n];
            }
        }
    }
#endif
}

void MM_ComputeBounded_reuseB_i8(const float * A,
                                 float * C,
                                 const uint8_t* W,
                                 const uint8_t* zp,
                                 const float* scales,
                                 int M, int IC, int OC,
                                 int n0, int n1) {
    constexpr int BK = 512;
    constexpr int BN = 512;
    auto bN_SIMDWx3 = BN / (SIMDW*3) * (SIMDW*3);
    auto bN_SIMDWx1 = BN - bN_SIMDWx3;
    float* scratch = scratch_alloc<float>(BN * BK + BN);
    float* repacked_B_n24 = scratch;
    float* repacked_B_n8 = repacked_B_n24 + bN_SIMDWx3 * BK;
    float* zero_points = repacked_B_n8 + SIMDW*3 * BK;

    const auto A_stride = IC;
    const auto B_stride = OC;
    const auto C_stride = OC;

    auto M_tails = M % 4;
    auto M_body = M - M_tails;

    for (int cur_n = n0; cur_n < n1; cur_n += BN) {
        int bN = std::min(n1 - cur_n, BN);
        const auto* pW = W + cur_n;

        // decompress zero-point
        if (zp) {
            for (int n = 0; n < bN; n += SIMDW) {
                auto zp0 = simd_load_epu8_epi32(static_cast<void const*>(zp + cur_n + n));
                auto zpf32 = simd_cvtepi32_ps(zp0);
                simd_storeu_ps(zero_points + n, zpf32);
            }
        }

        for (int k0 = 0; k0 < IC; k0 += BK, pW += BK * B_stride) {
            int bK = std::min(IC - k0, BK);
            if (zp) {
                repack_weight_for_4x3<true>(pW, B_stride,
                                    scales + cur_n,
                                    zero_points,
                                    bK, bN,
                                    repacked_B_n24,
                                    repacked_B_n8);
            } else {
                repack_weight_for_4x3<false>(pW, B_stride,
                                    scales + cur_n,
                                    zero_points,
                                    bK, bN,
                                    repacked_B_n24,
                                    repacked_B_n8);
            }

            bool is_accumulate_C = (k0 > 0);
            auto* pC = C + cur_n;
            int m;
            // re-use repacked B sub-matrix in L2 cache as long as we can.
            const auto* pA = A + k0;
            for (m = 0; m < M_body; m += 4, pA += 4 * A_stride, pC += 4 * C_stride) {
                auto* pB = repacked_B_n24;
                int n = 0;
                for (; n + SIMDW * 3 <= bN; n += SIMDW * 3, pB += SIMDW * 3 * bK)
                    brgemm_4x3<4>(pA, A_stride, pB, SIMDW*3, pC + n, C_stride, bK, is_accumulate_C);
                pB = repacked_B_n8;
                for (; n < bN; n += SIMDW, pB += SIMDW * bK)
                    brgemm_4x1<4>(pA, A_stride, pB, SIMDW, pC + n, C_stride, bK, is_accumulate_C);
            }
            // M tails
            for (; m < M; m++, pA += A_stride, pC += C_stride) {
                auto* pB = repacked_B_n24;
                int n = 0;
                for (; n + SIMDW * 3 <= bN; n += SIMDW * 3, pB += SIMDW * 3 * bK)
                    brgemm_4x3<1>(pA, A_stride, pB, SIMDW*3, pC + n, C_stride, bK, is_accumulate_C);
                pB = repacked_B_n8;
                for (; n < bN; n += SIMDW, pB += SIMDW * bK)
                    brgemm_4x1<1>(pA, A_stride, pB, SIMDW, pC + n, C_stride, bK, is_accumulate_C);
            }
        }
    }
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
    float* scratch = scratch_alloc<float>(BK * (SIMDW*2) + OC);

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
    float* zero_points = scratch + BK * (SIMDW*2);

    // deocompress zero-point into scratch
    if (zp) {
        int n = 0;
        for (n = n0; n + SIMDW <= n1; n += SIMDW) {
            auto vzpi32 = simd_load_epu8_epi32(static_cast<void const*>(zp + n));
            auto vzpf32 = simd_cvtepi32_ps(vzpi32);
            simd_storeu_ps(zero_points + n - n0, vzpf32);
        }
        for (; n < n1; n ++) {
            zero_points[n - n0] = zp[n];
        }
    }

    for (int k = 0; k < K; k += BK, A += BK, W += BK * W_stride) {
        int bK = std::min(K - k, BK);
        auto is_accumulate_C = (k > 0);

        for (int n = n0; n + 2 * SIMDW <= n1; n += 2 * SIMDW) {
            // prepack [BK, 16] into scratch
            if (zp) {
                auto* dst = repacked_B;
                SIMD_F32 vzp0;
                SIMD_F32 vzp1;
                vzp0 = simd_loadu_ps(zero_points + (n - n0));
                vzp1 = simd_loadu_ps(zero_points + (n - n0) + SIMDW);
                auto vscale0 = simd_loadu_ps(scales + n);
                auto vscale1 = simd_loadu_ps(scales + n + SIMDW);
                const auto* srcW = W + n;
                for (int k = 0; k < bK; k++, dst += 2 * SIMDW, srcW += W_stride) {
                    auto wb0 = simd_load_epu8_epi32(static_cast<void const*>(srcW + SIMDW * 0));
                    auto wb1 = simd_load_epu8_epi32(static_cast<void const*>(srcW + SIMDW * 1));
                    auto wf0 = simd_sub_ps(simd_cvtepi32_ps(wb0), vzp0);
                    auto wf1 = simd_sub_ps(simd_cvtepi32_ps(wb1), vzp1);
                    wf0 = simd_mul_ps(wf0, vscale0);
                    wf1 = simd_mul_ps(wf1, vscale1);

                    // prefetch right
                    simd_prefetch(srcW + 64, _MM_HINT_T0);

                    simd_storeu_ps(dst + SIMDW * 0, wf0);
                    simd_storeu_ps(dst + SIMDW * 1, wf1);
                }
            } else {
                auto* dst = repacked_B;
                auto vscale0 = simd_loadu_ps(scales + n);
                auto vscale1 = simd_loadu_ps(scales + n + SIMDW);
                const auto* srcW = W + n;
                for (int k = 0; k < bK; k++, dst += 2 * SIMDW, srcW += W_stride) {
                    auto wb0 = simd_load_epi8_epi32(static_cast<void const*>(srcW + SIMDW * 0));
                    auto wb1 = simd_load_epi8_epi32(static_cast<void const*>(srcW + SIMDW * 1));
                    auto wf0 = simd_cvtepi32_ps(wb0);
                    auto wf1 = simd_cvtepi32_ps(wb1);
                    wf0 = simd_mul_ps(wf0, vscale0);
                    wf1 = simd_mul_ps(wf1, vscale1);

                    // prefetch right
                    simd_prefetch(srcW + 64, _MM_HINT_T0);

                    simd_storeu_ps(dst + SIMDW * 0, wf0);
                    simd_storeu_ps(dst + SIMDW * 1, wf1);
                }
            }

            auto *pA = A;
            auto* pC = C + n;
            for (int m = 0; m < M_body; m += 6, pA += 6 * A_stride, pC += 6 * C_stride) {
                brgemm_6x2<6>(pA, A_stride, repacked_B, 2 * SIMDW, pC, C_stride, bK, is_accumulate_C);
            }
            if (brgmm6x2_tail)
                (*brgmm6x2_tail)(pA, A_stride, repacked_B, 2 * SIMDW, pC, C_stride, bK, is_accumulate_C);
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
    float* scratch = scratch_alloc<float>(BK * (SIMDW*2) + OC);

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
    float* zero_points = scratch + BK*(SIMDW*2);
    auto Z_stride = zp ? W_stride : 0;

    for (int k = 0; k < K; k += BK, A += BK, W += BK * W_stride, zp += Z_stride, scales += OC) {
        int bK = std::min(K - k, BK);
        auto is_accumulate_C = (k > 0);

        // deocompress zero-point into scratch buffer
        if (zp) {
            const auto* pzp = zp + n0/2;
            int n = 0;
            auto vmask_u4 = simd_set1_epi32(0xF);
            for (n = n0; n + SIMDW*2 <= n1; n += SIMDW*2, pzp += SIMDW) {
                auto vzp16xu4_i32 = simd_load_epu8_epi32(static_cast<void const*>(pzp));
                // 8 x low-4bits  : 0,1,2,3,4,5,6,7
                // 8 x high-4bits : 8,9,a,b,c,d,e,f

                auto vzp16xu4_i32_low = simd_and_si(vzp16xu4_i32, vmask_u4);
                auto vzp16xu4_i32_high = simd_srli_epi32(vzp16xu4_i32, 4);

                auto vzpf32_low = simd_cvtepi32_ps(vzp16xu4_i32_low);
                auto vzpf32_high = simd_cvtepi32_ps(vzp16xu4_i32_high);
                simd_storeu_ps(zero_points + n - n0, vzpf32_low);
                simd_storeu_ps(zero_points + n - n0 + SIMDW, vzpf32_high);
            }
            for (; n < n1; n += 2, pzp++) {
                zero_points[n - n0] = (*pzp) & 0xF;
                zero_points[n - n0 + 1] = (*pzp) >> 4;
            }
        }

        for (int n = n0; n + 2 * SIMDW <= n1; n += 2 * SIMDW) {
            // prepack subB [BK, 16] into scratch
            // because BK is fully contained within IC-group (BK == n*IC_group_size), it can share same zp & scales
            if (zp) {
                auto* dst = repacked_B;
                auto vzp0 = simd_loadu_ps(zero_points + (n - n0));
                auto vzp1 = simd_loadu_ps(zero_points + (n - n0) + SIMDW);
                auto vscale0 = simd_loadu_ps(scales + n);
                auto vscale1 = simd_loadu_ps(scales + n + SIMDW);
                const auto* srcW = W + n/2;
                auto vmask_u4 = simd_set1_epi32(0xF);
                for (int k = 0; k < bK; k++, dst += 2 * SIMDW, srcW += W_stride) {
                    // 16 x i4
                    auto wdw = simd_load_epu8_epi32(static_cast<void const*>(srcW + SIMDW * 0));

                    auto wdw_low = simd_and_si(wdw, vmask_u4);
                    auto wdw_high = simd_srli_epi32(wdw, 4);

                    auto wf0 = simd_sub_ps(simd_cvtepi32_ps(wdw_low), vzp0);
                    auto wf1 = simd_sub_ps(simd_cvtepi32_ps(wdw_high), vzp1);
                    wf0 = simd_mul_ps(wf0, vscale0);
                    wf1 = simd_mul_ps(wf1, vscale1);

                    // prefetch right
                    simd_prefetch(srcW + 64, _MM_HINT_T0);

                    simd_storeu_ps(dst + SIMDW * 0, wf0);
                    simd_storeu_ps(dst + SIMDW * 1, wf1);
                }
            } else {
                auto* dst = repacked_B;
                auto vscale0 = simd_loadu_ps(scales + n);
                auto vscale1 = simd_loadu_ps(scales + n + SIMDW);
                const auto* srcW = W + n/2;
                for (int k = 0; k < bK; k++, dst += 2 * SIMDW, srcW += W_stride) {
                    // 16 x i4
                    auto wdw = simd_load_epu8_epi32(static_cast<void const*>(srcW + SIMDW * 0));

                    auto wdw_low = simd_srai_epi32(simd_slli_epi32(wdw, 32-4), 32-4);
                    auto wdw_high = simd_srai_epi32(simd_slli_epi32(wdw, 32-8), 32-4);

                    auto wf0 = simd_cvtepi32_ps(wdw_low);
                    auto wf1 = simd_cvtepi32_ps(wdw_high);
                    wf0 = simd_mul_ps(wf0, vscale0);
                    wf1 = simd_mul_ps(wf1, vscale1);

                    // prefetch right
                    simd_prefetch(srcW + 64, _MM_HINT_T0);

                    simd_storeu_ps(dst + SIMDW * 0, wf0);
                    simd_storeu_ps(dst + SIMDW * 1, wf1);
                }
            }

            // matmul(A, subB)
            auto *pA = A;
            auto* pC = C + n;
            for (int m = 0; m < M_body; m += 6, pA += 6 * A_stride, pC += 6 * C_stride) {
                brgemm_6x2<6>(pA, A_stride, repacked_B, 2 * SIMDW, pC, C_stride, bK, is_accumulate_C);
            }
            if (brgmm6x2_tail)
                (*brgmm6x2_tail)(pA, A_stride, repacked_B, 2 * SIMDW, pC, C_stride, bK, is_accumulate_C);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void dynPruneLinear_i8(const float* input,      // [M, IC]
                        float threshold,
                        float zero_point,
                        const uint8_t* W,       // [IC, OC]
                        const uint8_t* zp,      // [OC]
                        const float* scales,    // [OC]
                        float* output,          // [M, OC]
                        int M, int IC, int OC) {
    if (M > 1) {
        if (M < 32) {
            parallel_nt(0, [&](const int ithr, const int nthr) {
                int n0, n1;
                splitter(OC/(2*SIMDW), nthr, ithr, n0, n1);
                n0 *= 2*SIMDW;
                n1 *= 2*SIMDW;
                MM_ComputeBounded_reuseA_i8(
                    input, output,
                    W, zp, scales, M, IC, OC, n0, n1);
            });
        } else {
            parallel_nt(0, [&](const int ithr, const int nthr) {
                int n0, n1;
                splitter(OC/(SIMDW), nthr, ithr, n0, n1);
                n0 *= SIMDW;
                n1 *= SIMDW;
                MM_ComputeBounded_reuseB_i8(
                    input, output,
                    W, zp, scales, M, IC, OC, n0, n1);
            });
        }
        return;
    }

    auto prof = PROFILE("gate_ids");
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
    prof = PROFILE("mm");

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
        if (zp)
            accumulate_w8_peroc<true>(pdst, OC, &gate_ids[g0], g1 - g0, W, zp, scales, &gate_val[g0], IC);
        else
            accumulate_w8_peroc<false>(pdst, OC, &gate_ids[g0], g1 - g0, W, zp, scales, &gate_val[g0], IC);
    });

    prof = PROFILE("reduce");
    reduce_outputs(output, output_temp.data(), nthr_max, M, OC);
    return;
}

void dynPruneLinear_i4(const float* input,      // [M, IC]
                        float threshold,
                        float zero_point,
                        const uint8_t* W,       // [IC, OC]
                        const uint8_t* zp,      // [OC]
                        const float* scales,    // [OC]
                        float* output,          // [M, OC]
                        int M, int IC, int OC,
                        int IC_group_size) {
    if ((OC % (2*SIMDW)) > 0) {
        throw std::runtime_error("OC is not multiple of 16");
    }

    if (M > 1) {
        // a reference impl
        parallel_nt(0, [&](const int ithr, const int nthr) {
            int n0, n1;
            splitter(OC/(2*SIMDW), nthr, ithr, n0, n1);
            n0 *= 2*SIMDW;
            n1 *= 2*SIMDW;
            MM_ComputeBounded_reuseA_i4(
                input, output,
                W, zp, scales, M, IC, OC, n0, n1, IC_group_size);
        });
        return;
    }

    auto prof = PROFILE("gate_ids");
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
    prof = PROFILE("mm");

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
        if (zp)
            accumulate_w4<true>(pdst, OC, &gate_ids[g0], g1 - g0, W, zp, scales, &gate_val[g0], IC, IC_group_size);
        else
            accumulate_w4<false>(pdst, OC, &gate_ids[g0], g1 - g0, W, zp, scales, &gate_val[g0], IC, IC_group_size);
    });

    prof = PROFILE("reduce");
    reduce_outputs(output, output_temp.data(), nthr_max, M, OC);
    return;
}

// [OC, IC/2, 2] => [IC, OC/2, 2]
// each row is further reordered in unit of 16 x i4 in [0,8,1,9,2,a,3,b,4,c,5,d,6,e,7,f] order
void dynPruneLinear_repack_i4(uint8_t * src, uint8_t * dst, int IC, int OC) {
    auto src_stride = IC / 2;

    int ic = 0;
    uint8_t scratch0[64];
    uint8_t scratch1[64];
    for (; ic + 2*SIMDW*4 <= IC; ic += 2*SIMDW*4) {
        // 64-ic
        auto* pdst = dst + ic * (OC / 2);
        auto vmask_low_u4 = simd_set1_epi8(0xF);
        auto vmask_high_u4 = simd_set1_epi8(0xF0);
        for (int oc = 0; oc < OC; oc += (SIMDW*2), pdst += SIMDW) {
            // 64-ic x 16-oc
            auto* psrc_oc0 = src + (ic / 2) + (oc + 0)*src_stride;
            auto* psrc_oc1 = src + (ic / 2) + (oc + SIMDW)*src_stride;
            for (int k = 0; k < SIMDW; k++, psrc_oc0 += src_stride, psrc_oc1 += src_stride) {
                auto b0 = simd_loadu_i32(psrc_oc0); // oc+0: ic0~64
                auto b1 = simd_loadu_i32(psrc_oc1); // oc+8: ic0~64
                auto b0_ic0 = simd_and_si(b0, vmask_low_u4);
                auto b0_ic1 = simd_and_si(simd_srli_epi16(b0, 4), vmask_low_u4);

                auto b1_ic0 = simd_and_si(simd_slli_epi16(b1, 4), vmask_high_u4);
                auto b1_ic1 = simd_and_si(b1, vmask_high_u4);

                auto bdst_ic0 = simd_or_si(b1_ic0, b0_ic0);    // even channels
                auto bdst_ic1 = simd_or_si(b1_ic1, b0_ic1);    // odd channels

                simd_storeu_si(scratch0, bdst_ic0);
                simd_storeu_si(scratch1, bdst_ic1);

                auto* pdst_temp0 = pdst + k;
                auto* pdst_temp1 = pdst + k + (OC / 2);
                for (int i = 0; i < SIMDW * 4; i++, pdst_temp0 += OC, pdst_temp1 += OC) {
                    *pdst_temp0 = scratch0[i];
                    *pdst_temp1 = scratch1[i];
                }
            }
        }
    }

    // [OC, IC/2, 2] => [IC, OC/2, 2]
    // tails
    for (; ic < IC; ic += 2) {
        auto* pdst_a = dst + ic * (OC / 2);
        auto* pdst_b = pdst_a + (OC / 2);
        for (int oc = 0; oc < OC; oc += SIMDW*2, pdst_a += SIMDW, pdst_b += SIMDW) {
            // interleave
            auto* psrc_oc0 = src + (ic / 2) + (oc + 0)*src_stride;
            auto* psrc_oc1 = src + (ic / 2) + (oc + SIMDW)*src_stride;
            for (int k = 0; k < SIMDW; k++, psrc_oc0 += src_stride, psrc_oc1 += src_stride) {
                auto data0 = *psrc_oc0;  // [ic1, ic0] packed in same u8
                auto u40a = (data0 & 0xF);
                auto u40b = (data0 >> 4);
                auto data1 = *psrc_oc1;
                auto u41a = (data1 & 0xF);
                auto u41b = (data1 >> 4);
                pdst_a[k] = (u41a << 4) | u40a;
                pdst_b[k] = (u41b << 4) | u40b;
            }
        }
    }
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov