// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "act_sparse_fc_kernel.hpp"

#include <cstring>

#include "openvino/core/except.hpp"

#if defined(OPENVINO_ARCH_X86_64)

#include "openvino/core/parallel.hpp"

namespace ov {
namespace intel_cpu {

enum class WeightCompressionType { FP16 = 0, INT8, INT4 };

static std::shared_ptr<SIMDJit> jit_compile_gemmRegBlk(int rows, int cols, int prefetch_B_adv = 0) {
    auto jit = std::make_shared<SIMDJit>(__func__);
    auto simd_width_bytes = SIMDJit::vmm_width<uint8_t>();

    auto is_preload_b = (rows >= cols);
    auto vregs_C = jit->get_vregs(rows * cols);
    auto vregs_B = jit->get_vregs(is_preload_b ? cols : 1);
    auto vregs_A = jit->get_vregs(is_preload_b ? 1 : rows);

    auto vmmC = [&](int row, int col) {
        return vregs_C[row * cols + col];
    };
    auto vmmB = [&](int col) {
        if (is_preload_b)
            return vregs_B[col];
        else
            return vregs_B[0];
    };
    auto vmmA = [&](int row) {
        if (is_preload_b)
            return vregs_A[0];
        else
            return vregs_A[row];
    };

    // load all arguments into register
    auto A_ptr = jit->get_arg(0);
    auto A_stride = jit->get_arg(1);
    auto B_ptr = jit->get_arg(2);
    auto B_stride = jit->get_arg(3);
    auto dst_ptr = jit->get_arg(4);
    auto dst_stride = jit->get_arg(5);
    auto K = jit->get_arg(6);
    auto accumulate = jit->get_arg(7);

    auto stemp = jit->get_sreg();

    A_stride = A_stride * 4;
    B_stride = B_stride * 4;
    dst_stride = dst_stride * 4;

    jit->if_(
        accumulate == 0,
        [&] {
            // initilaize C to zero
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++) {
                    auto ymm = vmmC(r, c);
                    jit->vxorps(ymm, ymm, ymm);
                }
        },
        [&] {
            // load subC[m_rows, m_cols]
            jit->mov(stemp, dst_ptr);
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    jit->simd_loadu_ps(vmmC(r, c), jit->ptr[stemp + c * simd_width_bytes]);
                }
                jit->add(stemp, dst_stride);
            }
        });

    // loop over K
    //            B:    1 x cols regs
    // A : 1 regs C: rows x cols regs
    auto A_ptr3 = accumulate;  // accumulate can be re-used
    auto loadA = [&](int r) {
        switch (r) {
        case 0:
            jit->simd_broadcast_ss(vmmA(r), jit->ptr[A_ptr + 0]);
            break;
        case 1:
            jit->simd_broadcast_ss(vmmA(r), jit->ptr[A_ptr + A_stride]);
            break;
        case 2:
            jit->simd_broadcast_ss(vmmA(r), jit->ptr[A_ptr + 2 * A_stride]);
            break;
        case 3:
            jit->simd_broadcast_ss(vmmA(r), jit->ptr[A_ptr3 + 0]);
            break;
        case 4:
            jit->simd_broadcast_ss(vmmA(r), jit->ptr[A_ptr3 + A_stride]);
            break;
        case 5:
            jit->simd_broadcast_ss(vmmA(r), jit->ptr[A_ptr3 + 2 * A_stride]);
            break;
        default:
            OPENVINO_ASSERT(false, "number of reg-blocking rows is not supported");
        }
    };

    if (rows > 3) {
        A_ptr3 = A_ptr + 3 * A_stride;
    }

    jit->do_while_(K > 0, [&]() {
        if (is_preload_b) {
            // preload B regs
            for (int c = 0; c < cols; c++)
                jit->simd_loadu_ps(vmmB(c), jit->ptr[B_ptr + c * simd_width_bytes]);

            if (prefetch_B_adv > 0)
                jit->prefetcht0(jit->ptr[B_ptr + prefetch_B_adv]);

            B_ptr = B_ptr + B_stride;
            for (int r = 0; r < rows; r++) {
                loadA(r);
                for (int c = 0; c < cols; c++)
                    jit->simd_fmadd_ps(vmmC(r, c), vmmA(r), vmmB(c));
            }

            A_ptr = A_ptr + 4;
            if (rows > 3)
                A_ptr3 = A_ptr3 + 4;
        } else {
            // preload A regs
            for (int r = 0; r < rows; r++)
                loadA(r);

            for (int c = 0; c < cols; c++) {
                jit->simd_loadu_ps(vmmB(c), jit->ptr[B_ptr + c * simd_width_bytes]);
                for (int r = 0; r < rows; r++)
                    jit->simd_fmadd_ps(vmmC(r, c), vmmA(r), vmmB(c));
            }

            B_ptr = B_ptr + B_stride;
            A_ptr = A_ptr + 4;
            if (rows > 3)
                A_ptr3 = A_ptr3 + 4;
        }
        K--;
    });

    // save C
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            jit->simd_storeu_ps(jit->ptr[dst_ptr + c * simd_width_bytes], vmmC(r, c));
        }
        jit->add(dst_ptr, dst_stride);
    }

    jit->finalize();
    return jit;
}

static void gemm6x2_Mx2(const float* pA,
                        int64_t A_stride,
                        const float* pB,
                        int64_t B_stride,
                        const float* pC,
                        int64_t C_stride,
                        int M,
                        int64_t bK,
                        int64_t is_accumulate_C) {
    static std::shared_ptr<SIMDJit> gemm6x2[6] = {
        jit_compile_gemmRegBlk(6, 2),
        jit_compile_gemmRegBlk(1, 2),
        jit_compile_gemmRegBlk(2, 2),
        jit_compile_gemmRegBlk(3, 2),
        jit_compile_gemmRegBlk(4, 2),
        jit_compile_gemmRegBlk(5, 2),
    };
    int m;
    for (m = 0; m + 6 <= M; m += 6, pA += 6 * A_stride, pC += 6 * C_stride) {
        (*gemm6x2[0])(pA, A_stride, pB, B_stride, pC, C_stride, bK, is_accumulate_C);
    }
    if (m < M)
        (*gemm6x2[M - m])(pA, A_stride, pB, B_stride, pC, C_stride, bK, is_accumulate_C);
}

static std::shared_ptr<SIMDJit> jit_compile_accumulate_weight_i4(bool with_zero_point) {
    auto jit = std::make_shared<SIMDJit>(__func__);
    auto simd_width = SIMDJit::vmm_width<float>();
    auto dst = jit->get_arg(0);          // float*
    auto p_w0 = jit->get_arg(1);         // int4*
    auto p_w1 = jit->get_arg(2);         // int4*
    auto p_w2 = jit->get_arg(3);         // int4*
    auto p_w3 = jit->get_arg(4);         // int4*
    auto dense_x = jit->get_arg(5);      // float*
    auto OC = jit->get_arg(6);           // float*
    auto scales = jit->get_arg(7);       // float*
    auto zero_points = jit->get_arg(8);  // float*

    auto oc = jit->get_sreg();

    auto vx = jit->get_vregs(4);

    auto vzp0 = jit->get_vreg();
    auto vzp1 = jit->get_vreg();

    auto vdst0 = jit->get_vreg();
    auto vdst1 = jit->get_vreg();

    auto vsum0 = jit->get_vreg();
    auto vsum1 = jit->get_vreg();

    auto wi32 = jit->get_vreg();
    auto wf0 = jit->get_vreg();
    auto wf1 = jit->get_vreg();

    auto vmask_u4 = jit->get_vreg();

    auto vscale0 = jit->get_vreg();
    auto vscale1 = jit->get_vreg();

    decltype(p_w0) pweights[4] = {p_w0, p_w1, p_w2, p_w3};

    jit->simd_set1_epi32(vmask_u4, 0xF);
    jit->simd_broadcast_ss(vx[0], jit->ptr[dense_x + 0 * sizeof(float)]);
    jit->simd_broadcast_ss(vx[1], jit->ptr[dense_x + 1 * sizeof(float)]);
    jit->simd_broadcast_ss(vx[2], jit->ptr[dense_x + 2 * sizeof(float)]);
    jit->simd_broadcast_ss(vx[3], jit->ptr[dense_x + 3 * sizeof(float)]);

    jit->for_loop(oc, 0, OC, simd_width * 2, [&]() {
        if (with_zero_point) {
            jit->simd_loadu_ps(vzp0, jit->ptr[zero_points + oc * sizeof(float) + 0 * simd_width * sizeof(float)]);
            jit->simd_loadu_ps(vzp1, jit->ptr[zero_points + oc * sizeof(float) + 1 * simd_width * sizeof(float)]);
        }

        jit->simd_loadu_ps(vdst0, jit->ptr[dst + oc * sizeof(float) + 0 * simd_width * sizeof(float)]);
        jit->simd_loadu_ps(vdst1, jit->ptr[dst + oc * sizeof(float) + 1 * simd_width * sizeof(float)]);

        jit->simd_loadu_ps(vscale0, jit->ptr[scales + oc * sizeof(float) + 0 * simd_width * sizeof(float)]);
        jit->simd_loadu_ps(vscale1, jit->ptr[scales + oc * sizeof(float) + 1 * simd_width * sizeof(float)]);

        jit->simd_setzero_ps(vsum0);
        jit->simd_setzero_ps(vsum1);

        for (int ic = 0; ic < 4; ic++) {
            jit->simd_load_epu8_epi32(wi32, jit->ptr[pweights[ic] + 0]);
            pweights[ic] = pweights[ic] + simd_width;

            if (with_zero_point) {
                // u4->i32
                jit->simd_and(wf0, wi32, vmask_u4);
                jit->simd_srli_epi32(wf1, wi32, 4);
            } else {
                // i4->i32
                jit->simd_slli_epi32(wf0, wi32, 32 - 4);
                jit->simd_srai_epi32(wf0, wf0, 32 - 4);
                jit->simd_slli_epi32(wf1, wi32, 32 - 8);
                jit->simd_srai_epi32(wf1, wf1, 32 - 4);
            }

            jit->simd_cvtepi32_ps(wf0, wf0);
            jit->simd_cvtepi32_ps(wf1, wf1);
            if (with_zero_point) {
                jit->simd_sub_ps(wf0, wf0, vzp0);
                jit->simd_sub_ps(wf1, wf1, vzp1);
            }

            jit->simd_fmadd_ps(vsum0, wf0, vx[ic]);
            jit->simd_fmadd_ps(vsum1, wf1, vx[ic]);
        }
        jit->simd_fmadd_ps(vdst0, vsum0, vscale0);
        jit->simd_fmadd_ps(vdst1, vsum1, vscale1);
        jit->simd_storeu_ps(jit->ptr[dst + oc * sizeof(float) + 0 * simd_width * sizeof(float)], vdst0);
        jit->simd_storeu_ps(jit->ptr[dst + oc * sizeof(float) + 1 * simd_width * sizeof(float)], vdst1);
    });

    jit->finalize(oc);
    return jit;
}
static std::shared_ptr<SIMDJit> jit_compile_accumulate_weight(WeightCompressionType wtype, bool with_zp = false) {
    auto jit = std::make_shared<SIMDJit>(__func__);
    auto simd_width = SIMDJit::vmm_width<float>();
    // load all arguments into register
    auto dst = jit->get_arg(0);  // float*
    auto OC = jit->get_arg(1);
    auto gate_ids = jit->get_arg(2);     // int32_t *
    auto gate_cnt = jit->get_arg(3);     // int
    auto pw0 = jit->get_arg(4);          // ov::float16* / uint8_t*
    auto dense_x = jit->get_arg(5);      //
    auto scales = jit->get_arg(6);       // float*
    auto zero_points = jit->get_arg(7);  // float*

    auto g = jit->get_sreg();
    auto i = jit->get_sreg();
    auto p_w0 = jit->get_sreg();
    auto p_w1 = jit->get_sreg();
    auto p_w2 = jit->get_sreg();
    auto p_w3 = jit->get_sreg();

    p_w0 = 0;
    p_w1 = 0;
    p_w2 = 0;
    p_w3 = 0;
    jit->for_loop(g, 0, gate_cnt, 4, [&]() {
        auto weight_element_size = (wtype == WeightCompressionType::FP16) ? 2 : 1;
        jit->mov(p_w0.r32(), jit->dword[gate_ids + g * 4 + 0 * 4]);
        jit->mov(p_w1.r32(), jit->dword[gate_ids + g * 4 + 1 * 4]);
        jit->mov(p_w2.r32(), jit->dword[gate_ids + g * 4 + 2 * 4]);
        jit->mov(p_w3.r32(), jit->dword[gate_ids + g * 4 + 3 * 4]);

        p_w0 = pw0 + p_w0 * OC * weight_element_size;
        p_w1 = pw0 + p_w1 * OC * weight_element_size;
        p_w2 = pw0 + p_w2 * OC * weight_element_size;
        p_w3 = pw0 + p_w3 * OC * weight_element_size;

        auto vx0 = jit->get_vreg();
        auto vx1 = jit->get_vreg();
        auto vx2 = jit->get_vreg();
        auto vx3 = jit->get_vreg();
        auto vscales = jit->get_vreg();
        auto vzp = jit->get_vreg();
        auto vdst = jit->get_vreg();
        auto vw0 = jit->get_vreg();
        auto vw1 = jit->get_vreg();
        auto vw2 = jit->get_vreg();
        auto vw3 = jit->get_vreg();
        auto vsum = jit->get_vreg();
        jit->simd_broadcast_ss(vx0, jit->ptr[dense_x + g * 4 + 0 * 4]);
        jit->simd_broadcast_ss(vx1, jit->ptr[dense_x + g * 4 + 1 * 4]);
        jit->simd_broadcast_ss(vx2, jit->ptr[dense_x + g * 4 + 2 * 4]);
        jit->simd_broadcast_ss(vx3, jit->ptr[dense_x + g * 4 + 3 * 4]);

        jit->for_loop(i, 0, OC, simd_width, [&]() {
            jit->simd_loadu_ps(vdst, jit->ptr[dst + i * 4]);
            if (wtype == WeightCompressionType::FP16) {
                jit->simd_loadu_phps(vw0, jit->ptr[p_w0 + i * 2]);
                jit->simd_loadu_phps(vw1, jit->ptr[p_w1 + i * 2]);
                jit->simd_loadu_phps(vw2, jit->ptr[p_w2 + i * 2]);
                jit->simd_loadu_phps(vw3, jit->ptr[p_w3 + i * 2]);
                jit->simd_fmadd_ps(vdst, vw0, vx0);
                jit->simd_fmadd_ps(vdst, vw1, vx1);
                jit->simd_fmadd_ps(vdst, vw2, vx2);
                jit->simd_fmadd_ps(vdst, vw3, vx3);
            } else if (wtype == WeightCompressionType::INT8) {
                jit->simd_setzero_ps(vsum);
                jit->simd_loadu_ps(vscales, jit->ptr[scales + i * 4]);
                if (with_zp) {
                    jit->simd_loadu_ps(vzp, jit->ptr[zero_points + i * 4]);
                    jit->simd_load_epu8_epi32(vw0, jit->ptr[p_w0 + i * 1]);
                    jit->simd_load_epu8_epi32(vw1, jit->ptr[p_w1 + i * 1]);
                    jit->simd_load_epu8_epi32(vw2, jit->ptr[p_w2 + i * 1]);
                    jit->simd_load_epu8_epi32(vw3, jit->ptr[p_w3 + i * 1]);
                    jit->simd_cvtepi32_ps(vw0, vw0);
                    jit->simd_cvtepi32_ps(vw1, vw1);
                    jit->simd_cvtepi32_ps(vw2, vw2);
                    jit->simd_cvtepi32_ps(vw3, vw3);
                    jit->simd_sub_ps(vw0, vw0, vzp);
                    jit->simd_sub_ps(vw1, vw1, vzp);
                    jit->simd_sub_ps(vw2, vw2, vzp);
                    jit->simd_sub_ps(vw3, vw3, vzp);
                } else {
                    jit->simd_load_epi8_epi32(vw0, jit->ptr[p_w0 + i * 1]);
                    jit->simd_load_epi8_epi32(vw1, jit->ptr[p_w1 + i * 1]);
                    jit->simd_load_epi8_epi32(vw2, jit->ptr[p_w2 + i * 1]);
                    jit->simd_load_epi8_epi32(vw3, jit->ptr[p_w3 + i * 1]);
                    jit->simd_cvtepi32_ps(vw0, vw0);
                    jit->simd_cvtepi32_ps(vw1, vw1);
                    jit->simd_cvtepi32_ps(vw2, vw2);
                    jit->simd_cvtepi32_ps(vw3, vw3);
                }
                jit->simd_fmadd_ps(vsum, vw0, vx0);
                jit->simd_fmadd_ps(vsum, vw1, vx1);
                jit->simd_fmadd_ps(vsum, vw2, vx2);
                jit->simd_fmadd_ps(vsum, vw3, vx3);
                jit->simd_fmadd_ps(vdst, vsum, vscales);
            }

            jit->simd_storeu_ps(jit->ptr[dst + i * 4], vdst);
        });
    });
    jit->finalize(i);
    return jit;
}

static std::shared_ptr<SIMDJit> jit_compile_reduce_outputs() {
    auto jit = std::make_shared<SIMDJit>(__func__);
    auto simd_width = SIMDJit::vmm_width<float>();
    // load all arguments into register
    auto dst0 = jit->get_arg(0);        // float*
    auto src0 = jit->get_arg(1);        // float*
    auto num_copies = jit->get_arg(2);  // int
    auto OC = jit->get_arg(3);          // int
    auto stride = jit->get_arg(4);      // int

    auto i = jit->get_sreg();
    auto k = jit->get_sreg();
    auto ptemp = jit->get_sreg();

    jit->for_loop(i, 0, OC, simd_width, [&] {
        ptemp = src0 + i * sizeof(float);
        auto vsum = jit->get_vreg();
        auto vw = jit->get_vreg();
        jit->simd_setzero_ps(vsum);
        jit->for_loop(k, 0, num_copies, 1, [&] {
            jit->simd_loadu_ps(vw, jit->ptr[ptemp + 0]);
            jit->simd_add_ps(vsum, vsum, vw);
            ptemp = ptemp + stride * sizeof(float);
        });
        jit->simd_storeu_ps(jit->ptr[dst0 + i * sizeof(float)], vsum);
    });

    jit->finalize();
    return jit;
}

/*
dst0 : [N, OC]
src0 : [num_copies, N, OC]
*/
static inline void reduce_outputs(float* dst0, float* src0, int num_copies, int64_t OC) {
    static auto jit_reduce = jit_compile_reduce_outputs();
    int64_t simd_width = SIMDJit::vmm_width<float>();

    ov::parallel_nt(0, [&](const int ithr, const int nthr) {
        int64_t oc0, oc1;
        ov::splitter(OC / simd_width, nthr, ithr, oc0, oc1);
        oc0 *= simd_width;
        oc1 *= simd_width;
        if (oc1 > OC)
            oc1 = OC;

        (*jit_reduce)(dst0 + oc0, src0 + oc0, num_copies, oc1 - oc0, OC);
    });
}

static std::shared_ptr<SIMDJit> jit_compile_repack_3xsimdw_1xsimdw(bool with_zp) {
    auto jit = std::make_shared<SIMDJit>(__func__);
    auto simd_width = SIMDJit::vmm_width<float>();
    // load all arguments into register
    auto src = jit->get_arg(0);             // uint8_t*
    auto strideW = jit->get_arg(1);         // int
    auto scales = jit->get_arg(2);          // float*
    auto zero_points = jit->get_arg(3);     // float*
    auto K = jit->get_arg(4);               // int
    auto N = jit->get_arg(5);               // int
    auto repacked_B_nx3 = jit->get_arg(6);  // float*
    auto repacked_B_nx1 = jit->get_arg(7);  // float*

    auto k = jit->get_sreg();
    auto n0 = jit->get_sreg();
    auto dst = jit->get_sreg();
    auto dst_stride = jit->get_sreg();

    auto wf0 = jit->get_vreg();
    auto wf1 = jit->get_vreg();
    auto wf2 = jit->get_vreg();

    auto scale0 = jit->get_vreg();
    auto scale1 = jit->get_vreg();
    auto scale2 = jit->get_vreg();

    auto zp0 = jit->get_vreg();
    auto zp1 = jit->get_vreg();
    auto zp2 = jit->get_vreg();

    jit->for_loop(k, 0, K, 1, [&]() {
        dst = repacked_B_nx3;
        dst_stride = K * simd_width * 3 * sizeof(float);

        jit->for_loop(n0, 0, N, simd_width * 3, [&]() {
            jit->simd_loadu_ps(scale0, jit->ptr[scales + n0 * sizeof(float) + 0 * simd_width * sizeof(float)]);
            jit->simd_loadu_ps(scale1, jit->ptr[scales + n0 * sizeof(float) + 1 * simd_width * sizeof(float)]);
            jit->simd_loadu_ps(scale2, jit->ptr[scales + n0 * sizeof(float) + 2 * simd_width * sizeof(float)]);
            if (with_zp) {
                jit->simd_loadu_ps(zp0, jit->ptr[zero_points + n0 * sizeof(float) + 0 * simd_width * sizeof(float)]);
                jit->simd_loadu_ps(zp1, jit->ptr[zero_points + n0 * sizeof(float) + 1 * simd_width * sizeof(float)]);
                jit->simd_loadu_ps(zp2, jit->ptr[zero_points + n0 * sizeof(float) + 2 * simd_width * sizeof(float)]);

                jit->simd_load_epu8_epi32(wf0, jit->ptr[src + n0 * sizeof(int8_t) + 0 * simd_width * sizeof(int8_t)]);
                jit->simd_load_epu8_epi32(wf1, jit->ptr[src + n0 * sizeof(int8_t) + 1 * simd_width * sizeof(int8_t)]);
                jit->simd_load_epu8_epi32(wf2, jit->ptr[src + n0 * sizeof(int8_t) + 2 * simd_width * sizeof(int8_t)]);
                jit->simd_cvtepi32_ps(wf0, wf0);
                jit->simd_cvtepi32_ps(wf1, wf1);
                jit->simd_cvtepi32_ps(wf2, wf2);
                jit->simd_sub_ps(wf0, wf0, zp0);
                jit->simd_sub_ps(wf1, wf1, zp1);
                jit->simd_sub_ps(wf2, wf2, zp2);
            } else {
                jit->simd_load_epi8_epi32(wf0, jit->ptr[src + n0 * sizeof(int8_t) + 0 * simd_width * sizeof(int8_t)]);
                jit->simd_load_epi8_epi32(wf1, jit->ptr[src + n0 * sizeof(int8_t) + 1 * simd_width * sizeof(int8_t)]);
                jit->simd_load_epi8_epi32(wf2, jit->ptr[src + n0 * sizeof(int8_t) + 2 * simd_width * sizeof(int8_t)]);
                jit->simd_cvtepi32_ps(wf0, wf0);
                jit->simd_cvtepi32_ps(wf1, wf1);
                jit->simd_cvtepi32_ps(wf2, wf2);
            }
            jit->simd_mul_ps(wf0, wf0, scale0);
            jit->simd_mul_ps(wf1, wf1, scale1);
            jit->simd_mul_ps(wf2, wf2, scale2);

            jit->simd_storeu_ps(jit->ptr[dst + 0 * simd_width * sizeof(float)], wf0);
            jit->simd_storeu_ps(jit->ptr[dst + 1 * simd_width * sizeof(float)], wf1);
            jit->simd_storeu_ps(jit->ptr[dst + 2 * simd_width * sizeof(float)], wf2);
            dst += dst_stride;
        });

        dst = repacked_B_nx1;
        dst_stride = K * (simd_width * 1 * sizeof(float));

        jit->for_loop(n0, n0, N, simd_width, [&]() {
            jit->simd_loadu_ps(scale0, jit->ptr[scales + n0 * sizeof(float)]);
            if (with_zp) {
                jit->simd_loadu_ps(zp0, jit->ptr[zero_points + n0 * sizeof(float)]);
                jit->simd_load_epu8_epi32(wf0, jit->ptr[src + n0 * sizeof(int8_t)]);
                jit->simd_cvtepi32_ps(wf0, wf0);
                jit->simd_sub_ps(wf0, wf0, zp0);
            } else {
                jit->simd_load_epi8_epi32(wf0, jit->ptr[src + n0 * sizeof(int8_t)]);
                jit->simd_cvtepi32_ps(wf0, wf0);
            }
            jit->simd_mul_ps(wf0, wf0, scale0);
            jit->simd_storeu_ps(jit->ptr[dst + 0], wf0);
            dst = dst + dst_stride;
        });
        // move to next row
        repacked_B_nx3 = repacked_B_nx3 + simd_width * 3 * sizeof(float);
        repacked_B_nx1 = repacked_B_nx1 + simd_width * 1 * sizeof(float);
        src = src + strideW;
    });
    jit->finalize();
    return jit;
}

static std::shared_ptr<SIMDJit> jit_compile_repack_2xsimdw(WeightCompressionType wtype, bool with_zero_point = false) {
    auto jit = std::make_shared<SIMDJit>(__func__);
    auto simd_width = SIMDJit::vmm_width<float>();
    // load all arguments into register
    auto src = jit->get_arg(0);         // pointer to ov::float16/u8/i8/i4
    auto src_stride = jit->get_arg(1);  // in unit of f16 or bytes (int8/int4)
    auto dst = jit->get_arg(2);         // float*
    auto bK = jit->get_arg(3);
    auto scales = jit->get_arg(4);      // scales
    auto zero_point = jit->get_arg(5);  // zero-point

    auto k = jit->get_sreg();

    auto wf0 = jit->get_vreg();
    auto wf1 = jit->get_vreg();

    auto vzp0 = jit->get_vreg();
    auto vzp1 = jit->get_vreg();

    auto vscale0 = jit->get_vreg();
    auto vscale1 = jit->get_vreg();

    auto wi4x2 = jit->get_vreg();
    auto vmask_u4 = jit->get_vreg();

    auto vtemp0 = jit->get_vreg();
    auto vtemp1 = jit->get_vreg();

    if (wtype == WeightCompressionType::INT4)
        jit->simd_set1_epi32(vmask_u4, 0xF);

    if (wtype == WeightCompressionType::INT8 || wtype == WeightCompressionType::INT4) {
        jit->simd_loadu_ps(vscale0, jit->ptr[scales + 0 * simd_width * sizeof(float)]);
        jit->simd_loadu_ps(vscale1, jit->ptr[scales + 1 * simd_width * sizeof(float)]);
        if (with_zero_point) {
            jit->simd_loadu_ps(vzp0, jit->ptr[zero_point + 0 * simd_width * sizeof(float)]);
            jit->simd_loadu_ps(vzp1, jit->ptr[zero_point + 1 * simd_width * sizeof(float)]);
        }
    }
    jit->for_loop(k, 0, bK, 1, [&]() {
        if (wtype == WeightCompressionType::FP16) {
            jit->simd_loadu_phps(wf0, jit->ptr[src + simd_width * 0 * sizeof(ov::float16)]);
            jit->simd_loadu_phps(wf1, jit->ptr[src + simd_width * 1 * sizeof(ov::float16)]);
        } else if (wtype == WeightCompressionType::INT8 || wtype == WeightCompressionType::INT4) {
            if (wtype == WeightCompressionType::INT4) {
                // 2xint4 is packed into a u8
                jit->simd_load_epu8_epi32(wi4x2, jit->ptr[src + simd_width * 0 * sizeof(uint8_t)]);
                if (with_zero_point) {
                    // uint4 => i32
                    jit->simd_and(wf0, wi4x2, vmask_u4);
                    jit->simd_srli_epi32(wf1, wi4x2, 4);
                } else {
                    // int4 => i32
                    jit->simd_slli_epi32(vtemp0, wi4x2, 32 - 4);
                    jit->simd_slli_epi32(vtemp1, wi4x2, 32 - 8);
                    jit->simd_srai_epi32(wf0, vtemp0, 32 - 4);
                    jit->simd_srai_epi32(wf1, vtemp1, 32 - 4);
                }
            } else {
                if (with_zero_point) {
                    jit->simd_load_epu8_epi32(wf0, jit->ptr[src + simd_width * 0 * sizeof(uint8_t)]);
                    jit->simd_load_epu8_epi32(wf1, jit->ptr[src + simd_width * 1 * sizeof(uint8_t)]);
                } else {
                    jit->simd_load_epi8_epi32(wf0, jit->ptr[src + simd_width * 0 * sizeof(uint8_t)]);
                    jit->simd_load_epi8_epi32(wf1, jit->ptr[src + simd_width * 1 * sizeof(uint8_t)]);
                }
            }

            jit->simd_cvtepi32_ps(wf0, wf0);
            jit->simd_cvtepi32_ps(wf1, wf1);
            if (with_zero_point) {
                jit->simd_sub_ps(wf0, wf0, vzp0);
                jit->simd_sub_ps(wf1, wf1, vzp1);
            }
            jit->simd_mul_ps(wf0, wf0, vscale0);
            jit->simd_mul_ps(wf1, wf1, vscale1);
        }

        jit->prefetcht0(jit->ptr[src + 64]);
        jit->simd_storeu_ps(jit->ptr[dst + simd_width * 0 * sizeof(float)], wf0);
        jit->simd_storeu_ps(jit->ptr[dst + simd_width * 1 * sizeof(float)], wf1);
        dst = dst + simd_width * 2 * sizeof(float);
        if (wtype == WeightCompressionType::FP16) {
            src = src + src_stride * sizeof(ov::float16);
        } else if (wtype == WeightCompressionType::INT8 || wtype == WeightCompressionType::INT4) {
            src = src + src_stride * sizeof(uint8_t);
        }
    });

    jit->finalize();
    return jit;
}

template <class T>
T* ActSparseFcKernel::scratch_alloc(size_t cnt) {
#    if defined(__GNUC__) || defined(__clang__)
    thread_local uint8_t scratch[1024 * 1024 * 2] __attribute__((aligned(4096)));
#    else
    thread_local uint8_t scratch[1024 * 1024 * 2];
#    endif
    OPENVINO_ASSERT(cnt * sizeof(T) < sizeof(scratch));
    // DEBUG_LOG(reinterpret_cast<void*>(scratch));
    return reinterpret_cast<T*>(scratch);
}

void ActSparseFcKernel::MM_ComputeBounded_reuseA_f16(const float* A,
                                                     float* C,
                                                     const ov::float16* W,
                                                     int M,
                                                     int IC,
                                                     int OC,
                                                     int n0,
                                                     int n1) {
    static auto repack_2xsimdw = jit_compile_repack_2xsimdw(WeightCompressionType::FP16);
    constexpr int BK = 54;
    const auto SIMDW = SIMDJit::vmm_width<float>();
    float* scratch = scratch_alloc<float>(BK * (SIMDW * 2) + OC);

    int K = IC;
    int64_t A_stride = IC;
    int64_t C_stride = OC;
    int64_t W_stride = OC;

    float* repacked_B = scratch;

    for (int k = 0; k < K; k += BK, A += BK, W += BK * W_stride) {
        int64_t bK = std::min(K - k, BK);
        int64_t is_accumulate_C = (k > 0);

        for (int n = n0; n + 2 * SIMDW <= n1; n += 2 * SIMDW) {
            // prepack [BK, 16] into scratch
            (*repack_2xsimdw)(W + n, W_stride, repacked_B, bK);
            gemm6x2_Mx2(A, A_stride, repacked_B, 2 * SIMDW, C + n, C_stride, M, bK, is_accumulate_C);
        }
    }
}

static std::shared_ptr<SIMDJit> get_decompress_zp_u8() {
    auto jit = std::make_shared<SIMDJit>(__func__);
    auto simd_width = SIMDJit::vmm_width<float>();

    auto zp_input_u8 = jit->get_arg(0);
    auto zp_output_f32 = jit->get_arg(1);
    auto cnt = jit->get_arg(2);

    auto n = jit->get_sreg();

    auto vzpi32 = jit->get_vreg();
    jit->for_loop(n, 0, cnt, simd_width, [&]() {
        jit->simd_load_epu8_epi32(vzpi32, jit->ptr[zp_input_u8 + n * 1]);
        jit->simd_cvtepi32_ps(vzpi32, vzpi32);
        jit->simd_storeu_ps(jit->ptr[zp_output_f32 + n * 4], vzpi32);
    });
    // tails are converted using C instead.
    jit->finalize(n);
    return jit;
}

static std::shared_ptr<SIMDJit> get_decompress_zp_u4() {
    auto jit = std::make_shared<SIMDJit>(__func__);
    auto simd_width = SIMDJit::vmm_width<float>();

    auto zp_input_u4 = jit->get_arg(0);
    auto zp_output_f32 = jit->get_arg(1);
    auto cnt = jit->get_arg(2);

    auto n = jit->get_sreg();

    auto vzpi4x2 = jit->get_vreg();
    auto vzpi32_lo = jit->get_vreg();
    auto vzpi32_hi = jit->get_vreg();
    auto vmask_u4 = jit->get_vreg();

    jit->simd_set1_epi32(vmask_u4, 0xF);

    jit->for_loop(n, 0, cnt, simd_width * 2, [&]() {
        jit->simd_load_epu8_epi32(vzpi4x2, jit->ptr[zp_input_u4 + 0]);
        zp_input_u4 = zp_input_u4 + simd_width;
        jit->simd_and(vzpi32_lo, vzpi4x2, vmask_u4);
        jit->simd_srli_epi32(vzpi32_hi, vzpi4x2, 4);
        jit->simd_cvtepi32_ps(vzpi32_lo, vzpi32_lo);
        jit->simd_cvtepi32_ps(vzpi32_hi, vzpi32_hi);
        jit->simd_storeu_ps(jit->ptr[zp_output_f32 + n * sizeof(float)], vzpi32_lo);
        jit->simd_storeu_ps(jit->ptr[zp_output_f32 + n * sizeof(float) + simd_width * sizeof(float)], vzpi32_hi);
    });
    // tails are converted using C instead.
    jit->finalize(n);
    return jit;
}

void ActSparseFcKernel::MM_ComputeBounded_reuseA_i8(const float* A,
                                                    float* C,
                                                    const uint8_t* W,
                                                    const uint8_t* zp,
                                                    const float* scales,
                                                    int M,
                                                    int IC,
                                                    int OC,
                                                    int64_t n0,
                                                    int64_t n1) {
    static auto decompress_zp_u8 = get_decompress_zp_u8();
    static auto repack_2xsimdw_i8_zp = jit_compile_repack_2xsimdw(WeightCompressionType::INT8, true);
    static auto repack_2xsimdw_i8_nozp = jit_compile_repack_2xsimdw(WeightCompressionType::INT8, false);

    auto repack_2xsimdw_i8 = zp ? repack_2xsimdw_i8_zp : repack_2xsimdw_i8_nozp;
    constexpr int BK = 54;
    const auto SIMDW = SIMDJit::vmm_width<float>();
    float* scratch = scratch_alloc<float>(BK * (SIMDW * 2) + OC);

    int K = IC;
    auto A_stride = IC;
    auto C_stride = OC;
    auto W_stride = OC;

    float* repacked_B = scratch;
    float* zero_points = scratch + BK * (SIMDW * 2);

    // deocompress zero-point into scratch
    if (zp) {
        int n = n0 + (*decompress_zp_u8)(zp + n0, zero_points, n1 - n0);
        for (; n < n1; n++) {
            zero_points[n - n0] = zp[n];
        }
    }

    for (int k = 0; k < K; k += BK, A += BK, W += BK * W_stride) {
        int bK = std::min(K - k, BK);
        auto is_accumulate_C = (k > 0);

        for (int n = n0; n + 2 * SIMDW <= n1; n += 2 * SIMDW) {
            // prepack [BK, 16] into scratch
            (*repack_2xsimdw_i8)(W + n, W_stride, repacked_B, bK, scales + n, zero_points + n - n0);
            gemm6x2_Mx2(A, A_stride, repacked_B, 2 * SIMDW, C + n, C_stride, M, bK, is_accumulate_C);
        }
    }
}

void ActSparseFcKernel::MM_ComputeBounded_reuseB_i8(const float* A,
                                                    float* C,
                                                    const uint8_t* W,
                                                    const uint8_t* zp,
                                                    const float* scales,
                                                    int M,
                                                    int IC,
                                                    int OC,
                                                    int n0,
                                                    int n1) {
    static auto decompress_zp_u8 = get_decompress_zp_u8();
    static std::shared_ptr<SIMDJit> gemm4x3[6] = {
        jit_compile_gemmRegBlk(4, 3),
        jit_compile_gemmRegBlk(1, 3),
        jit_compile_gemmRegBlk(2, 3),
        jit_compile_gemmRegBlk(3, 3),
    };
    static std::shared_ptr<SIMDJit> gemm4x1[6] = {
        jit_compile_gemmRegBlk(4, 1),
        jit_compile_gemmRegBlk(1, 1),
        jit_compile_gemmRegBlk(2, 1),
        jit_compile_gemmRegBlk(3, 1),
    };

    static auto repack_3xsimdw_i8_1xsimdw_nozp = jit_compile_repack_3xsimdw_1xsimdw(false);
    static auto repack_3xsimdw_i8_1xsimdw_withzp = jit_compile_repack_3xsimdw_1xsimdw(true);
    auto* repack_3xsimdw_i8 = zp ? repack_3xsimdw_i8_1xsimdw_withzp.get() : repack_3xsimdw_i8_1xsimdw_nozp.get();

    const auto SIMDW = SIMDJit::vmm_width<float>();
    constexpr int BK = 512;
    constexpr int BN = 512;
    auto bN_SIMDWx3 = BN / (SIMDW * 3) * (SIMDW * 3);
    float* scratch = scratch_alloc<float>(BN * BK + BN);
    float* repacked_B_n24 = scratch;
    float* repacked_B_n8 = repacked_B_n24 + bN_SIMDWx3 * BK;
    float* zero_points = repacked_B_n8 + SIMDW * 3 * BK;

    const int64_t A_stride = IC;
    const int64_t B_stride = OC;
    const int64_t C_stride = OC;

    for (int cur_n = n0; cur_n < n1; cur_n += BN) {
        int bN = std::min(n1 - cur_n, BN);
        const auto* pW = W + cur_n;

        if (zp) {
            (*decompress_zp_u8)(zp + cur_n, zero_points, bN);
        }

        for (int k0 = 0; k0 < IC; k0 += BK, pW += BK * B_stride) {
            int64_t bK = std::min(IC - k0, BK);
            (*repack_3xsimdw_i8)(pW, B_stride, scales + cur_n, zero_points, bK, bN, repacked_B_n24, repacked_B_n8);

            bool is_accumulate_C = (k0 > 0);
            auto* pC = C + cur_n;
            int m;
            // re-use repacked B sub-matrix in L2 cache as long as we can.
            const auto* pA = A + k0;
            for (m = 0; m + 4 <= M; m += 4, pA += 4 * A_stride, pC += 4 * C_stride) {
                auto* pB = repacked_B_n24;
                int n = 0;
                for (; n + SIMDW * 3 <= bN; n += SIMDW * 3, pB += SIMDW * 3 * bK)
                    (*gemm4x3[0])(pA, A_stride, pB, SIMDW * 3, pC + n, C_stride, bK, is_accumulate_C);
                pB = repacked_B_n8;
                for (; n < bN; n += SIMDW, pB += SIMDW * bK)
                    (*gemm4x1[0])(pA, A_stride, pB, SIMDW, pC + n, C_stride, bK, is_accumulate_C);
            }
            // M tails
            if (m < M) {
                auto* pB = repacked_B_n24;
                int n = 0;
                for (; n + SIMDW * 3 <= bN; n += SIMDW * 3, pB += SIMDW * 3 * bK)
                    (*gemm4x3[M - m])(pA, A_stride, pB, SIMDW * 3, pC + n, C_stride, bK, is_accumulate_C);
                pB = repacked_B_n8;
                for (; n < bN; n += SIMDW, pB += SIMDW * bK)
                    (*gemm4x1[M - m])(pA, A_stride, pB, SIMDW, pC + n, C_stride, bK, is_accumulate_C);
            }
        }
    }
}

void ActSparseFcKernel::MM_ComputeBounded_reuseA_i4(const float* A,
                                                    float* C,
                                                    const uint8_t* W,
                                                    const uint8_t* zp,
                                                    const float* scales,
                                                    int M,
                                                    int IC,
                                                    int OC,
                                                    int n0,
                                                    int n1,
                                                    int icgs) {
    static auto decompress_zp_u4 = get_decompress_zp_u4();
    static auto repack_2xsimdw_nozp = jit_compile_repack_2xsimdw(WeightCompressionType::INT4, false);
    static auto repack_2xsimdw_withzp = jit_compile_repack_2xsimdw(WeightCompressionType::INT4, true);
    auto* repack_2xsimdw = zp ? repack_2xsimdw_withzp.get() : repack_2xsimdw_nozp.get();

    int BK = icgs;
    const auto SIMDW = SIMDJit::vmm_width<float>();
    float* scratch = scratch_alloc<float>(BK * (SIMDW * 2) + OC);

    int K = IC;
    auto A_stride = IC;
    auto C_stride = OC;
    auto W_stride = (OC / 2);

    float* repacked_B = scratch;
    float* zero_points = scratch + BK * (SIMDW * 2);
    auto Z_stride = zp ? W_stride : 0;

    for (int k = 0; k < K; k += BK, A += BK, W += BK * W_stride, zp += Z_stride, scales += OC) {
        int bK = std::min(K - k, BK);
        auto is_accumulate_C = (k > 0);

        // deocompress zero-point into scratch buffer
        if (zp) {
            const auto* pzp = zp + n0 / 2;
            int n = n0 + (*decompress_zp_u4)(pzp, zero_points, n1 - n0);
            for (; n < n1; n += 2, pzp++) {
                zero_points[n - n0] = (*pzp) & 0xF;
                zero_points[n - n0 + 1] = (*pzp) >> 4;
            }
        }

        for (int n = n0; n + 2 * SIMDW <= n1; n += 2 * SIMDW) {
            // prepack subB [BK, 16] into scratch
            // because BK is fully contained within IC-group (BK == n*IC_group_size), it can share same zp & scales
            (*repack_2xsimdw)(W + n / 2, W_stride, repacked_B, bK, scales + n, zero_points + n - n0);
            gemm6x2_Mx2(A, A_stride, repacked_B, 2 * SIMDW, C + n, C_stride, M, bK, is_accumulate_C);
        }
    }
}

// [OC, IC/2, 2] => [IC, OC/2, 2]
// each row is further reordered in unit of 16 x i4 in [0,8,1,9,2,a,3,b,4,c,5,d,6,e,7,f] order
void ActSparseFcKernel::repack_weights_i4(uint8_t* src, uint8_t* dst, int IC, int OC) {
    auto src_stride = IC / 2;
    const auto SIMDW = SIMDJit::vmm_width<float>();

    // [OC, IC/2, 2] => [IC, OC/2, 2]
    // tails
    parallel_nt(0, [&](const int ithr, const int nthr) {
        int oc0, oc1;
        splitter(OC / (2 * SIMDW), nthr, ithr, oc0, oc1);
        oc0 *= 2 * SIMDW;
        oc1 *= 2 * SIMDW;

        for (int ic = 0; ic < IC; ic += 2) {
            auto* pdst_a = dst + ic * (OC / 2) + oc0 / 2;
            auto* pdst_b = pdst_a + (OC / 2);

            for (int oc = oc0; oc < oc1; oc += SIMDW * 2, pdst_a += SIMDW, pdst_b += SIMDW) {
                // interleave
                auto* psrc_oc0 = src + (ic / 2) + (oc + 0) * src_stride;
                auto* psrc_oc1 = src + (ic / 2) + (oc + SIMDW) * src_stride;
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
    });
}

ActSparseFcKernel::ActSparseFcKernel(bool is_quantized, bool is_int4, bool with_zero_points, int ic_group_size)
    : m_is_quantized(is_quantized),
      m_is_int4(is_int4),
      m_with_zp(with_zero_points),
      m_ic_group_size(ic_group_size) {
    static auto decompress_zp_u8 = get_decompress_zp_u8();
    static auto decompress_zp_u4 = get_decompress_zp_u4();

    static auto accumulate_weight_fp16 = jit_compile_accumulate_weight(WeightCompressionType::FP16);
    static auto accumulate_weight_i8_nozp = jit_compile_accumulate_weight(WeightCompressionType::INT8, false);
    static auto accumulate_weight_i8_withzp = jit_compile_accumulate_weight(WeightCompressionType::INT8, true);
    static auto accumulate_weight_i4_nozp = jit_compile_accumulate_weight_i4(false);
    static auto accumulate_weight_i4_withzp = jit_compile_accumulate_weight_i4(true);

    WeightCompressionType wtype;
    if (m_is_quantized)
        wtype = (m_is_int4 ? WeightCompressionType::INT4 : WeightCompressionType::INT8);
    else
        wtype = WeightCompressionType::FP16;

    switch (wtype) {
    case WeightCompressionType::FP16:
        m_accumulate_kernel = accumulate_weight_fp16.get();
        m_decompzp_kernel = nullptr;
        break;
    case WeightCompressionType::INT8:
        m_accumulate_kernel = m_with_zp ? accumulate_weight_i8_withzp.get() : accumulate_weight_i8_nozp.get();
        m_decompzp_kernel = decompress_zp_u8.get();
        break;
    case WeightCompressionType::INT4:
        m_accumulate_kernel = m_with_zp ? accumulate_weight_i4_withzp.get() : accumulate_weight_i4_nozp.get();
        m_decompzp_kernel = decompress_zp_u4.get();
        break;
    }
}

void ActSparseFcKernel::operator()(const float* input,
                                   float* output,
                                   int M,
                                   int IC,
                                   int OC,
                                   float threshold,
                                   float zero_point,
                                   const void* W,
                                   const float* scales,
                                   const uint8_t* zp) {
    const auto SIMDW = SIMDJit::vmm_width<float>();
    OPENVINO_ASSERT((OC % (2 * SIMDW)) == 0, "ActSparseFcKernel: OC is not multiple of ", 2 * SIMDW);

    WeightCompressionType wtype;
    if (m_is_quantized)
        wtype = (m_is_int4 ? WeightCompressionType::INT4 : WeightCompressionType::INT8);
    else
        wtype = WeightCompressionType::FP16;

    if (M > 1) {
        const auto SIMDW = SIMDJit::vmm_width<float>();
        if (wtype == WeightCompressionType::FP16) {
            parallel_nt(0, [&](const int ithr, const int nthr) {
                int n0, n1;
                splitter(OC / (2 * SIMDW), nthr, ithr, n0, n1);
                n0 *= 2 * SIMDW;
                n1 *= 2 * SIMDW;
                MM_ComputeBounded_reuseA_f16(input, output, reinterpret_cast<const ov::float16*>(W), M, IC, OC, n0, n1);
            });
            return;
        }
        if (wtype == WeightCompressionType::INT8) {
            if (M < 32) {
                parallel_nt(0, [&](const int ithr, const int nthr) {
                    int n0, n1;
                    splitter(OC / (2 * SIMDW), nthr, ithr, n0, n1);
                    n0 *= 2 * SIMDW;
                    n1 *= 2 * SIMDW;
                    MM_ComputeBounded_reuseA_i8(input,
                                                output,
                                                reinterpret_cast<const uint8_t*>(W),
                                                zp,
                                                scales,
                                                M,
                                                IC,
                                                OC,
                                                n0,
                                                n1);
                });
            } else {
                parallel_nt(0, [&](const int ithr, const int nthr) {
                    int n0, n1;
                    splitter(OC / (SIMDW), nthr, ithr, n0, n1);
                    n0 *= SIMDW;
                    n1 *= SIMDW;
                    MM_ComputeBounded_reuseB_i8(input,
                                                output,
                                                reinterpret_cast<const uint8_t*>(W),
                                                zp,
                                                scales,
                                                M,
                                                IC,
                                                OC,
                                                n0,
                                                n1);
                });
            }
            return;
        }
        if (wtype == WeightCompressionType::INT4) {
            parallel_nt(0, [&](const int ithr, const int nthr) {
                int n0, n1;
                splitter(OC / (2 * SIMDW), nthr, ithr, n0, n1);
                n0 *= 2 * SIMDW;
                n1 *= 2 * SIMDW;
                MM_ComputeBounded_reuseA_i4(input,
                                            output,
                                            reinterpret_cast<const uint8_t*>(W),
                                            zp,
                                            scales,
                                            M,
                                            IC,
                                            OC,
                                            n0,
                                            n1,
                                            m_ic_group_size);
            });
            return;
        }
        return;
    }

    // collect non-zero(non-sparse) activation channel id & value
    m_nonzero_cnt = 0;
    m_nonzero_ids.resize(IC);
    m_nonzero_val.resize(IC);
    auto IC_group_size = m_ic_group_size > 0 ? m_ic_group_size : IC;
    for (int c0 = 0; c0 < IC; c0 += IC_group_size) {
        for (int c1 = 0; c1 < IC_group_size; c1++) {
            auto channel = c0 + c1;
            auto& value = input[channel];
            if (std::abs(value - zero_point) >= threshold) {
                m_nonzero_ids[m_nonzero_cnt] = channel;
                m_nonzero_val[m_nonzero_cnt] = value;
                m_nonzero_cnt++;
            }
        }
        if (m_nonzero_cnt & 3) {
            // padding : ensuer 4 rows are from same group
            auto n_pad = 4 - (m_nonzero_cnt & 3);
            auto ic_pad = m_nonzero_ids[m_nonzero_cnt - 1];
            for (int i = 0; i < n_pad; i++) {
                m_nonzero_ids[m_nonzero_cnt] = ic_pad;
                m_nonzero_val[m_nonzero_cnt] = 0.0f;
                m_nonzero_cnt++;
            }
        }
    }

    auto nthr_max = parallel_get_max_threads();
    m_output_temp.resize(nthr_max * OC);

    thread_local std::vector<float> zpbuff;

    parallel_nt(0, [&](const int ithr, const int nthr) {
        int g0, g1;
        splitter(m_nonzero_cnt / 4, nthr, ithr, g0, g1);
        g0 *= 4;
        g1 *= 4;
        auto* pdst = &m_output_temp[ithr * OC];
        memset(pdst, 0, OC * sizeof(m_output_temp[0]));

        if (wtype == WeightCompressionType::FP16) {
            (*m_accumulate_kernel)(pdst, OC, &m_nonzero_ids[g0], (g1 - g0), W, &m_nonzero_val[g0]);
        } else if (wtype == WeightCompressionType::INT8) {
            if (zp) {
                zpbuff.resize(OC);
                (*m_decompzp_kernel)(zp, zpbuff.data(), OC);
            }
            (*m_accumulate_kernel)(pdst, OC, &m_nonzero_ids[g0], g1 - g0, W, &m_nonzero_val[g0], scales, zpbuff.data());
        } else if (wtype == WeightCompressionType::INT4) {
            zpbuff.resize(OC);
            int last_gid = -1;
            // vector x weights
            for (int g = g0; g < g1; g += 4) {
                auto ic0 = m_nonzero_ids[g];
                auto ic1 = m_nonzero_ids[g + 1];
                auto ic2 = m_nonzero_ids[g + 2];
                auto ic3 = m_nonzero_ids[g + 3];
                auto gid = ic0 / m_ic_group_size;
                auto* p_scales = scales + gid * OC;

                // entering a new group, decompress zero-points
                if (last_gid != gid) {
                    if (zp)
                        (*m_decompzp_kernel)(zp + gid * (OC / 2), zpbuff.data(), OC);
                    last_gid = gid;
                }

                const auto* p_w0 = reinterpret_cast<const uint8_t*>(W) + ic0 * OC / 2;
                const auto* p_w1 = reinterpret_cast<const uint8_t*>(W) + ic1 * OC / 2;
                const auto* p_w2 = reinterpret_cast<const uint8_t*>(W) + ic2 * OC / 2;
                const auto* p_w3 = reinterpret_cast<const uint8_t*>(W) + ic3 * OC / 2;
                (*m_accumulate_kernel)(pdst, p_w0, p_w1, p_w2, p_w3, &m_nonzero_val[g], OC, p_scales, zpbuff.data());
            }
        }
    });
    reduce_outputs(output, m_output_temp.data(), nthr_max, OC);
}

}  // namespace intel_cpu
}  // namespace ov

#endif  // OPENVINO_ARCH_X86_64
