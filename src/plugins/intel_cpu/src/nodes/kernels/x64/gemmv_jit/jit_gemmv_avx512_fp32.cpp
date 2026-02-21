// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "jit_gemmv_avx512_fp32.hpp"

#include <cstring>
#include <cmath>
#include "openvino/core/except.hpp"
#include "jit_prebuilt_pool.hpp"
#include "xbyak/xbyak_util.h"
#include <immintrin.h>

namespace ov::intel_cpu::x64::gemmv_jit {

// Local enums mirror to avoid pulling headers into JIT
static inline int to_int(quant_granularity_t q) { return static_cast<int>(q); }

// Intrinsics VNNI GEMV block without repack: processes one M-block (<=16 rows)
// W layout: interleave_m16, i.e., for each k: 16 bytes for rows [m..m+15]
#if defined(__GNUC__)
__attribute__((target("avx512vnni,avx512bw,avx512f")))
#endif
static inline void vnni_norepack_block_u8s8_to_fp32(
        const uint8_t* xq, int K,
        const uint8_t* wblk, int ld_w_bytes,
        const float* scales, const int32_t* zps, const float* bias,
        float* y, int M_valid,
        quant_granularity_t gran,
        float s_x, int32_t zp_x, int32_t sum_x_q) {
    const int M_blk = 16;
    const int full = M_valid >= M_blk ? M_blk : M_valid;
    __m512i acc = _mm512_setzero_si512();
    __m512i sumw = _mm512_setzero_si512();
    const __m512i maskB0 = _mm512_set1_epi32(0x000000FF);
    const __m512i maskB1 = _mm512_set1_epi32(0x0000FF00);
    const __m512i maskB2 = _mm512_set1_epi32(0x00FF0000);
    const __m512i maskB3 = _mm512_set1_epi32(0xFF000000u);
    for (int k = 0; k < K; k += 4) {
        for (int t = 0; t < 4; ++t) {
            if (k + t >= K) break;
            const uint8_t* wk = wblk + (size_t)(k + t) * 16;
            __m128i wb = _mm_loadu_si128((const __m128i*)wk);
            __m256i lo = _mm256_cvtepi8_epi32(wb);
            wb = _mm_srli_si128(wb, 8);
            __m256i hi = _mm256_cvtepi8_epi32(wb);
            __m512i w32 = _mm512_castsi256_si512(lo);
            w32 = _mm512_inserti64x4(w32, hi, 1);
            // sumW if needed (when zp or zp_x compensation is present)
            sumw = _mm512_add_epi32(sumw, w32);
            // place weight byte into byte t of each dword
            if (t == 1) w32 = _mm512_slli_epi32(w32, 8);
            else if (t == 2) w32 = _mm512_slli_epi32(w32, 16);
            else if (t == 3) w32 = _mm512_slli_epi32(w32, 24);
            __m512i m = (t == 0) ? maskB0 : (t == 1) ? maskB1 : (t == 2) ? maskB2 : maskB3;
            w32 = _mm512_and_si512(w32, m);
            const uint32_t xb = (uint32_t)xq[k + t] * 0x01010101u;
            __m512i xbr = _mm512_set1_epi32((int)xb);
            acc = _mm512_dpbusd_epi32(acc, xbr, w32);
        }
    }
    // compensation: acc += (-zp_x)*sumW
    if (zp_x != 0) {
        __m512i neg_xzp = _mm512_set1_epi32(-zp_x);
        __m512i tmp = _mm512_mullo_epi32(neg_xzp, sumw);
        acc = _mm512_add_epi32(acc, tmp);
    }
    // acc += zp_w * (K*zp_x - sum_x_q)
    if (zps) {
        const int c = K * zp_x - sum_x_q;
        __m512i cv = _mm512_set1_epi32(c);
        if (gran == quant_granularity_t::per_tensor) {
            __m512i zw = _mm512_set1_epi32(zps[0]);
            zw = _mm512_mullo_epi32(zw, cv);
            acc = _mm512_add_epi32(acc, zw);
        } else {
            __m512i zw;
            if (full < 16) {
                __mmask16 k1 = (__mmask16)((1u << full) - 1u);
                zw = _mm512_maskz_loadu_epi32(k1, zps);
            } else {
                zw = _mm512_loadu_si512((const void*)zps);
            }
            zw = _mm512_mullo_epi32(zw, cv);
            acc = _mm512_add_epi32(acc, zw);
        }
    }
    // to fp32 and scale by s_w*s_x
    __m512 yf = _mm512_cvtepi32_ps(acc);
    __m512 s;
    if (gran == quant_granularity_t::per_tensor) {
        s = _mm512_set1_ps(scales[0]);
    } else {
        if (full < 16) {
            __mmask16 k1 = (__mmask16)((1u << full) - 1u);
            s = _mm512_maskz_loadu_ps(k1, scales);
        } else {
            s = _mm512_loadu_ps(scales);
        }
    }
    const __m512 sx = _mm512_set1_ps(s_x);
    s = _mm512_mul_ps(s, sx);
    yf = _mm512_mul_ps(yf, s);
    // add bias
    if (bias) {
        __m512 b;
        if (gran == quant_granularity_t::per_tensor) b = _mm512_set1_ps(bias[0]);
        else {
            if (full < 16) { __mmask16 k1 = (__mmask16)((1u << full) - 1u); b = _mm512_maskz_loadu_ps(k1, bias); }
            else b = _mm512_loadu_ps(bias);
        }
        yf = _mm512_add_ps(yf, b);
    }
    // store (respect tail)
    if (full < 16) {
        __mmask16 k1 = (__mmask16)((1u << full) - 1u);
        _mm512_mask_storeu_ps(y, k1, yf);
    } else {
        _mm512_storeu_ps(y, yf);
    }
}

jit_gemmv_avx512_fp32_kernel::jit_gemmv_avx512_fp32_kernel()
    : dnnl::impl::cpu::x64::jit_generator_t("jit_gemmv_avx512_fp32_kernel",
            dnnl::impl::cpu::x64::cpu_isa_t::avx512_core) {
    auto st = create_kernel();
    if (st != dnnl::impl::status::success) {
        OPENVINO_THROW("Failed to build jit_gemmv_avx512_fp32 kernel");
    }
    fn_ = reinterpret_cast<fn_t>(jit_ker());
}

// JIT generator body
void jit_gemmv_avx512_fp32_kernel::generate() {
    using namespace Xbyak;
    this->setDefaultJmpNEAR(true);
#if defined(OPENVINO_ARCH_X86_64)
    endbr64();
#endif

    // Registers
    const Reg64 reg_args = rdi;
    const Reg64 reg_x = r8;
    const Reg64 reg_w = r9;
    const Reg64 reg_sc = r10;
    const Reg64 reg_zp = r11;
    const Reg64 reg_bias = r12;
    const Reg64 reg_y = r13;
    const Reg64 reg_k = r14;
    const Reg64 reg_k_iter = r15;

    // no masks in simplified path

    // Zmm registers
    const Zmm zmmC = zmm0;
    const Zmm zmmW = zmm1;
    const Zmm zmmX = zmm2;
    const Zmm zmmS = zmm3;
    const Zmm zmmZP = zmm4;      // zp (fp32)
    const Zmm zmmZPComp = zmm5;  // -s*zp*sum_x
    const Zmm zmmTmp = zmm6;
    const Xmm xmmWbytes = xmm7;  // weights buffer (8 or 16 bytes per k)
    const Ymm ymmIlo = ymm8;     // int32 low half
    const Ymm ymmIhi = ymm9;     // int32 high half
    const Ymm ymmFlo = ymm10;    // fp32 low half
    const Ymm ymmFhi = ymm11;    // fp32 high half
    // INT4 helpers
    const Ymm ymmWw   = ymm12;   // widened bytes -> words
    const Ymm ymmTmpW = ymm13;   // temp words
    const Ymm ymmMask000F = ymm14; // word mask 0x000F
    const Xmm xmmMask0F = Xbyak::Xmm(1); // byte mask 0x0F (temp for setup only)

    // Prologue: save callee-saved regs we use (System V ABI)
    push(r12);
    push(r13);
    push(r14);
    push(r15);

    // Load args
    mov(reg_x, ptr[reg_args + offsetof(gemmv_avx512_fp32_call_args, x)]);
    mov(reg_w, ptr[reg_args + offsetof(gemmv_avx512_fp32_call_args, wq)]);
    mov(reg_sc, ptr[reg_args + offsetof(gemmv_avx512_fp32_call_args, scales)]);
    mov(reg_zp, ptr[reg_args + offsetof(gemmv_avx512_fp32_call_args, zps)]);
    mov(reg_bias, ptr[reg_args + offsetof(gemmv_avx512_fp32_call_args, bias)]);
    mov(reg_y, ptr[reg_args + offsetof(gemmv_avx512_fp32_call_args, y)]);
    mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, K)]);
    mov(reg_k, rax);

    // Prepare constants for INT4 decode
    {
        // xmmMask0F = 0x0F (per byte); ymmMask000F = 0x000F (per word)
        mov(eax, 0x0F0F0F0F);
        movd(xmmMask0F, eax);
        pshufd(xmmMask0F, xmmMask0F, 0);
        vpmovzxbw(ymmMask000F, xmmMask0F);
        // 0x0008 word constant will be materialized on demand in the loop
    }

    // Load scales vector
    // if per_tensor: broadcast s[0]; else load 16 floats with tail mask if needed
    {
        Label L_not_pt, L_after_s;
        mov(eax, ptr[reg_args + offsetof(gemmv_avx512_fp32_call_args, gran)]);
        cmp(eax, to_int(quant_granularity_t::per_tensor));
        jne(L_not_pt);
        // per_tensor
        vbroadcastss(zmmS, dword[reg_sc]);
        jmp(L_after_s);
        L(L_not_pt);
        {
            Label L_s_full, L_s_done;
            mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, M_tail)]);
            test(eax, eax);
            jz(L_s_full);
            // k1 mask for tail
            mov(ecx, eax);
            mov(eax, 1);
            shl(eax, cl);
            sub(eax, 1);
            kmovw(k1, eax);
            vmovups(zmmS | k1 | T_z, ptr[reg_sc]);
            jmp(L_s_done);
            L(L_s_full);
            vmovups(zmmS, ptr[reg_sc]);
            L(L_s_done);
        }
        L(L_after_s);
    }

    // Optional VNNI path: branch early if enabled
    {
        Label L_no_vnni;
        mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, use_vnni)]);
        test(eax, eax);
        jz(L_no_vnni);

        // ---------------- VNNI (u8 X) x (s8 W) -> s32 accumulation ----------------
        // Registers reuse: zmmC as fp32 output later; use zAcc/zSumW for s32 accum
        const Zmm zAcc = zmm0;
        const Zmm zSumW = zmm1;
        const Zmm zF0  = zmm2;     // fp32 tmp for final scaling
        const Zmm zTmp  = zmm3;     // tmp
        const Zmm zXbr  = zmm4;     // broadcasted x byte into dwords
        const Zmm zW32  = zmm5;     // s32 lanes from W bytes (expanded)
        const Zmm zMask = zmm6;     // dword mask FF << (j*8)
        const Zmm zOnes = zmm7;     // bytes=1 repeated in all positions (as dwords 0x01010101)
        const Zmm zSsx  = zmm8;     // scales * s_x
        const Zmm zZP   = zmm9;     // zps as int32 -> fp32 (in comp stage)
        const Zmm zComp = zmm10;    // compensation in fp32
        const Xmm xWb   = xmm11;    // 16 weight bytes per k-step
        const Ymm yIlo  = ymm12;
        const Ymm yIhi  = ymm13;

        // Prepare zOnes = 0x01010101
        mov(eax, 0x01010101);
        movd(xmm0, eax);
        vpbroadcastd(zOnes, xmm0);

        // Initialize accumulators
        vpxord(zAcc, zAcc, zAcc);
        vpxord(zSumW, zSumW, zSumW);

        // Precompute zSsx = zmmS * s_x
        vbroadcastss(zTmp, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, s_x)]);
        vmulps(zSsx, zmmS, zTmp);

        // K loop in groups of up to 4 (handle tail)
        xor_(reg_k_iter, reg_k_iter);
        Label L_k_loop_vnni, L_k_done_vnni;
        L(L_k_loop_vnni);
        cmp(reg_k_iter, reg_k);
        jge(L_k_done_vnni);
        {
            // Process up to 4 k-steps
            for (int j = 0; j < 4; ++j) {
                Label L_step_done;
                // if (k_iter >= K) break
                cmp(reg_k_iter, reg_k);
                jge(L_step_done);
                // Load 16 weight bytes for this k
                vmovdqu8(xWb, ptr[reg_w]);
                add(reg_w, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, k_step_bytes)]);
                // Expand to 32-bit lanes (signed)
                vpmovsxbd(yIlo, xWb);
                vpsrldq(xWb, xWb, 8);
                vpmovsxbd(yIhi, xWb);
                vxorps(zW32, zW32, zW32);
                vinsertf32x8(zW32, zW32, yIlo, 0);
                vinsertf32x8(zW32, zW32, yIhi, 1);

                // SumW accumulation if needed
                {
                    Label L_no_sumw;
                    mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, need_sumw)]);
                    test(eax, eax); jz(L_no_sumw);
                    vpaddd(zSumW, zSumW, zW32);
                    L(L_no_sumw);
                }

                // Position bytes in j-th byte of each dword: (zW32 << (j*8)) & (0xFF<<(j*8))
                if (j != 0) { vpslld(zW32, zW32, j * 8); }
                mov(eax, 0xFF);
                movd(xmm0, eax);
                vpbroadcastd(zMask, xmm0);
                if (j != 0) { vpslld(zMask, zMask, j * 8); }
                vpandd(zW32, zW32, zMask);

                // Broadcast x_q8 byte and replicate to all bytes of each dword
                // eax = xq[k_iter + j]
                mov(rax, ptr[reg_args + offsetof(gemmv_avx512_fp32_call_args, x_q8)]);
                movzx(eax, byte[rax + reg_k_iter]);
                add(reg_k_iter, 1);
                // eax *= 0x01010101 to replicate byte across dword
                imul(eax, eax, 0x01010101);
                movd(xmm0, eax);
                vpbroadcastd(zXbr, xmm0);

                // Accumulate: acc += dpbusd(x_broadcast, placed_weights)
                vpdpbusd(zAcc, zXbr, zW32);

                L(L_step_done);
            }
        }
        jmp(L_k_loop_vnni);
        L(L_k_done_vnni);

        // Compensation terms
        // acc += (-zp_x) * sumW
        {
            Label L_skip_sumw;
            mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, need_sumw)]);
            test(eax, eax); jz(L_skip_sumw);
            mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, zp_x)]);
            neg(eax);
            movd(xmm0, eax);
            vpbroadcastd(zTmp, xmm0);          // zTmp = -zp_x (int32)
            vpmulld(zTmp, zTmp, zSumW);        // zTmp = (-zp_x) * sumW
            vpaddd(zAcc, zAcc, zTmp);
            L(L_skip_sumw);
        }

        // acc += zp_w * (K*zp_x - sum_x_q)
        {
            Label L_no_zp;
            mov(rax, reg_zp);
            test(rax, rax); jz(L_no_zp);
            // c = K*zp_x - sum_x_q
            mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, zp_x)]);
            imul(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, K)]);
            sub(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, sum_x_q)]);
            movd(xmm0, eax);
            vpbroadcastd(zTmp, xmm0);          // zTmp = c (int32)

            // per_tensor vs vector
            Label L_pt, L_vec, L_done_comp;
            mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, gran)]);
            cmp(eax, to_int(quant_granularity_t::per_tensor)); jne(L_vec);
            // per_tensor: scalar zp_w
            mov(eax, dword[reg_zp]);
            movd(xmm1, eax);
            vpbroadcastd(zZP, xmm1);
            vpmulld(zZP, zZP, zTmp);
            vpaddd(zAcc, zAcc, zZP);
            jmp(L_done_comp);
            L(L_vec);
            {
                Label L_zp_full, L_zp_done;
                mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, M_tail)]);
                test(eax, eax); jz(L_zp_full);
                mov(ecx, eax); mov(eax, 1); shl(eax, cl); sub(eax, 1); kmovw(k1, eax);
                vmovdqu32(zZP | k1 | T_z, ptr[reg_zp]);
                jmp(L_zp_done);
                L(L_zp_full);
                vmovdqu32(zZP, ptr[reg_zp]);
                L(L_zp_done);
            }
            vpmulld(zZP, zZP, zTmp);
            vpaddd(zAcc, zAcc, zZP);
            L(L_done_comp);
            L(L_no_zp);
        }

        // Convert to fp32 and scale: y = float(acc) * (s_w * s_x)
        vcvtdq2ps(zF0, zAcc);
        vmulps(zF0, zF0, zSsx);

        // Bias add
        {
            Label L_no_bias2, L_bias_pt2, L_bias_done2;
            mov(rax, reg_bias);
            test(rax, rax); jz(L_no_bias2);
            mov(eax, ptr[reg_args + offsetof(gemmv_avx512_fp32_call_args, gran)]);
            cmp(eax, to_int(quant_granularity_t::per_tensor)); jne(L_bias_pt2);
            vbroadcastss(zTmp, dword[reg_bias]);
            vaddps(zF0, zF0, zTmp);
            jmp(L_bias_done2);
            L(L_bias_pt2);
            {
                Label L_b_full2, L_b_done2;
                mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, M_tail)]);
                test(eax, eax); jz(L_b_full2);
                mov(ecx, eax); mov(eax, 1); shl(eax, cl); sub(eax, 1); kmovw(k1, eax);
                vmovups(zTmp | k1 | T_z, ptr[reg_bias]);
                jmp(L_b_done2);
                L(L_b_full2);
                vmovups(zTmp, ptr[reg_bias]);
                L(L_b_done2);
            }
            vaddps(zF0, zF0, zTmp);
            L(L_bias_done2);
            L(L_no_bias2);
        }

        // Gate and activation (ReLU) then store
        {
            Label L_no_gate2;
            mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, fuse_gate)]);
            test(eax, eax); jz(L_no_gate2);
            vbroadcastss(zTmp, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, gate)]);
            vmulps(zF0, zF0, zTmp);
            L(L_no_gate2);
        }
        {
            Label L_no_act2;
            mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, act_kind)]);
            cmp(eax, 1); jne(L_no_act2);
            vxorps(zTmp, zTmp, zTmp);
            vmaxps(zF0, zF0, zTmp);
            L(L_no_act2);
        }
        {
            Label L_full2, L_store_done2;
            mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, M_tail)]);
            test(eax, eax); jz(L_full2);
            mov(ecx, eax); mov(eax, 1); shl(eax, cl); sub(eax, 1); kmovw(k1, eax);
            vmovups(ptr[reg_y] | k1, zF0);
            jmp(L_store_done2);
            L(L_full2);
            vmovups(ptr[reg_y], zF0);
            L(L_store_done2);
        }

        // Epilogue: restore and return
        pop(r15);
        pop(r14);
        pop(r13);
        pop(r12);
        ret();

        L(L_no_vnni);
    }

    // bias accum init: zmmC = accumulate? load(y) : 0
    {
        Label L_acc, L_init_done;
        mov(eax, ptr[reg_args + offsetof(gemmv_avx512_fp32_call_args, accumulate)]);
        test(eax, eax);
        jz(L_acc);
        // accumulate==1 -> load
        vmovups(zmmC, ptr[reg_y]);
        jmp(L_init_done);
        L(L_acc);
        vxorps(zmmC, zmmC, zmmC);
        L(L_init_done);
    }

    // optional zp compensation: zmmZPComp = -s * zp * sum_x
    {
        Label L_no_zp, L_after_zp;
        mov(rax, reg_zp);
        test(rax, rax);
        jz(L_no_zp);
        // load zp -> int32 -> fp32
        // if per_tensor: broadcast; else load with tail mask if needed
        Label L_not_pt_zp, L_zp_cont;
        mov(eax, ptr[reg_args + offsetof(gemmv_avx512_fp32_call_args, gran)]);
        cmp(eax, to_int(quant_granularity_t::per_tensor));
        jne(L_not_pt_zp);
        {
            mov(eax, dword[reg_zp]);
            movd(xmm0, eax);
            vpbroadcastd(zmmZP, xmm0);
            jmp(L_zp_cont);
        }
        L(L_not_pt_zp);
        {
            Label L_zp_full, L_zp_done;
            mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, M_tail)]);
            test(eax, eax);
            jz(L_zp_full);
            mov(ecx, eax);
            mov(eax, 1);
            shl(eax, cl);
            sub(eax, 1);
            kmovw(k1, eax);
            vmovdqu32(zmmZP | k1 | T_z, ptr[reg_zp]);
            jmp(L_zp_done);
            L(L_zp_full);
            vmovdqu32(zmmZP, ptr[reg_zp]);
            L(L_zp_done);
        }
        L(L_zp_cont);
        vcvtdq2ps(zmmZP, zmmZP);
        vbroadcastss(zmmZPComp, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, sum_x)]);
        vmulps(zmmZPComp, zmmZPComp, zmmS);    // s * sum_x
        vmulps(zmmZPComp, zmmZPComp, zmmZP);   // s * sum_x * zp
        vsubps(zmmC, zmmC, zmmZPComp);         // C -= s*zp*sum_x
        jmp(L_after_zp);
        L(L_no_zp);
        // no zp -> nothing
        L(L_after_zp);
    }

    // K loop
    xor_(reg_k_iter, reg_k_iter);
    Label L_k_loop, L_k_done;
    L(L_k_loop);
    cmp(reg_k_iter, reg_k);
    jge(L_k_done);

    // decode weights: int8/u8 or int4/u4
    {
        Label L_bits8, L_after_decode;
        mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, w_nbits)]);
        cmp(eax, 8);
        je(L_bits8);
        // --- 4-bit path ---
        // load 8 bytes per k-step
        prefetcht0(ptr[reg_w + 64]);
        vmovq(xmmWbytes, ptr[reg_w]);
        add(reg_w, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, k_step_bytes)]);
        // widen to words (8->8)
        Xbyak::Xmm xA = Xbyak::Xmm(8);    // expanded words
        Xbyak::Xmm xB = Xbyak::Xmm(11);   // mask 0x000F words
        Xbyak::Xmm xZ = Xbyak::Xmm(15);   // zero
        vpxor(xZ, xZ, xZ);
        vpmovzxbw(xA, xmmWbytes);                 // xA = words(B0..B7)
        vextracti128(xB, ymmMask000F, 0);         // xB = 0x000F mask
        // low nibble -> xLo8
        Xbyak::Xmm xLow = Xbyak::Xmm(9);
        vmovdqa(xLow, xA);
        vpand(xLow, xLow, xB);
        Xbyak::Xmm xLo8 = Xbyak::Xmm(12);
        vpackuswb(xLo8, xLow, xZ);
        // high nibble -> xHi8
        Xbyak::Xmm xHigh = Xbyak::Xmm(10);
        vmovdqa(xHigh, xA);
        vpsrlw(xHigh, xHigh, 4);
        vpand(xHigh, xHigh, xB);
        Xbyak::Xmm xHi8 = Xbyak::Xmm(13);
        vpackuswb(xHi8, xHigh, xZ);
        // interleave bytes: [lo0,hi0, ... lo7,hi7] into 16B
        vmovdqa(xmmWbytes, xLo8);
        punpcklbw(xmmWbytes, xHi8);
        // note: i4 sign adjust is applied later on dword lanes
        // bytes -> dwords -> (optional i4 adjust) -> fp32 (two halves)
        vpmovzxbd(ymmIlo, xmmWbytes);
        vpsrldq(xmmWbytes, xmmWbytes, 8);
        vpmovzxbd(ymmIhi, xmmWbytes);
        {
            Label L_u4_cvt, L_after_cvt4;
            mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, w_unsigned)]);
            test(eax, eax);
            jnz(L_u4_cvt);
            // i4: apply (q ^ 8) - 8 on dword lanes
            Xbyak::Ymm ymmC8 = ymm13; // temp
            mov(eax, 8);
            Xbyak::Xmm xTmp = Xbyak::Xmm(12);
            movd(xTmp, eax);
            pshufd(xTmp, xTmp, 0);
            vpbroadcastd(ymmC8, xTmp);
            vpxor(ymmIlo, ymmIlo, ymmC8);
            vpsubd(ymmIlo, ymmIlo, ymmC8);
            vpxor(ymmIhi, ymmIhi, ymmC8);
            vpsubd(ymmIhi, ymmIhi, ymmC8);
            L(L_u4_cvt);
            L(L_after_cvt4);
        }
        vcvtdq2ps(ymmFlo, ymmIlo);
        vcvtdq2ps(ymmFhi, ymmIhi);
        // assemble zmmW
        vxorps(zmmW, zmmW, zmmW);
        vinsertf32x8(zmmW, zmmW, ymmFlo, 0);
        vinsertf32x8(zmmW, zmmW, ymmFhi, 1);
        jmp(L_after_decode);
        // --- 8-bit path ---
        L(L_bits8);
        // software prefetch for the next k-step
        prefetcht0(ptr[reg_w + 64]);
        vmovdqu(xmmWbytes, ptr[reg_w]);
        add(reg_w, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, k_step_bytes)]);
        {
            Label L_u8, L_after_cvt8;
            mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, w_unsigned)]);
            test(eax, eax);
            jz(L_u8);
            // u8: zero-extend via ymm halves
            vpmovzxbd(ymmIlo, xmmWbytes);
            vcvtdq2ps(ymmFlo, ymmIlo);
            vpsrldq(xmmWbytes, xmmWbytes, 8);
            vpmovzxbd(ymmIhi, xmmWbytes);
            vcvtdq2ps(ymmFhi, ymmIhi);
            vxorps(zmmW, zmmW, zmmW);
            vinsertf32x8(zmmW, zmmW, ymmFlo, 0);
            vinsertf32x8(zmmW, zmmW, ymmFhi, 1);
            jmp(L_after_cvt8);
            L(L_u8);
            // i8: sign-extend
            vpmovsxbd(ymmIlo, xmmWbytes);
            vcvtdq2ps(ymmFlo, ymmIlo);
            vpsrldq(xmmWbytes, xmmWbytes, 8);
            vpmovsxbd(ymmIhi, xmmWbytes);
            vcvtdq2ps(ymmFhi, ymmIhi);
            vxorps(zmmW, zmmW, zmmW);
            vinsertf32x8(zmmW, zmmW, ymmFlo, 0);
            vinsertf32x8(zmmW, zmmW, ymmFhi, 1);
            L(L_after_cvt8);
        }
        L(L_after_decode);
    }

    // debug capture: if enabled and k==dbg_k, store q (pre-scale)
    {
        Label L_no_dbg, L_after_dbg, L_no_q, L_no_qs;
        mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, dbg_enable)]);
        test(eax, eax);
        jz(L_no_dbg);
        // compare k
        mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, dbg_k)]);
        cmp(eax, reg_k_iter.cvt32());
        jne(L_no_dbg);
        // store q pre-scale if dbg_q provided
        mov(rax, ptr[reg_args + offsetof(gemmv_avx512_fp32_call_args, dbg_q)]);
        test(rax, rax); jz(L_no_q);
        vmovups(ptr[rax], zmmW);
        L(L_no_q);
        L(L_after_dbg);
        L(L_no_dbg);
    }

    // dequant: w_real = w * s (bias applied later)
    vmulps(zmmW, zmmW, zmmS);

    // debug capture: if enabled and k==dbg_k, store q*s
    {
        Label L_no_dbg2, L_after_dbg2, L_no_qs;
        mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, dbg_enable)]);
        test(eax, eax);
        jz(L_no_dbg2);
        mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, dbg_k)]);
        cmp(eax, reg_k_iter.cvt32());
        jne(L_no_dbg2);
        // store q*s if dbg_qs provided
        mov(rax, ptr[reg_args + offsetof(gemmv_avx512_fp32_call_args, dbg_qs)]);
        test(rax, rax); jz(L_no_qs);
        vmovups(ptr[rax], zmmW);
        L(L_no_qs);
        L(L_after_dbg2);
        L(L_no_dbg2);
    }

    // load & broadcast x[k]
    vbroadcastss(zmmX, dword[reg_x + reg_k_iter * 4]);
    // C += W * X
    vfmadd231ps(zmmC, zmmW, zmmX);

    inc(reg_k_iter);
    jmp(L_k_loop);
    L(L_k_done);

    // Add bias if provided
    {
        Label L_no_bias, L_bias_pt, L_bias_done;
        mov(rax, reg_bias);
        test(rax, rax);
        jz(L_no_bias);
        // bias per_tensor vs per_channel
        mov(eax, ptr[reg_args + offsetof(gemmv_avx512_fp32_call_args, gran)]);
        cmp(eax, to_int(quant_granularity_t::per_tensor));
        jne(L_bias_pt);
        // per_tensor: broadcast bias[0]
        vbroadcastss(zmmTmp, dword[reg_bias]);
        vaddps(zmmC, zmmC, zmmTmp);
        jmp(L_bias_done);
        L(L_bias_pt);
        // per_channel: load vector (masked if tail) then add
        {
            Label L_b_full, L_b_done;
            mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, M_tail)]);
            test(eax, eax);
            jz(L_b_full);
            mov(ecx, eax);
            mov(eax, 1);
            shl(eax, cl);
            sub(eax, 1);
            kmovw(k1, eax);
            vmovups(zmmTmp | k1 | T_z, ptr[reg_bias]);
            jmp(L_b_done);
            L(L_b_full);
            vmovups(zmmTmp, ptr[reg_bias]);
            L(L_b_done);
        }
        vaddps(zmmC, zmmC, zmmTmp);
        L(L_bias_done);
        L(L_no_bias);
    }

    // Store Y with optional M-tail mask
    {
        using namespace Xbyak;
        // Optional gate multiply: y *= gate
        {
            Label L_no_gate;
            mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, fuse_gate)]);
            test(eax, eax); jz(L_no_gate);
            vbroadcastss(zmmTmp, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, gate)]);
            vmulps(zmmC, zmmC, zmmTmp);
            L(L_no_gate);
        }
        // Optional activation: act_kind==1 -> ReLU
        {
            Label L_no_act;
            mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, act_kind)]);
            cmp(eax, 1); jne(L_no_act);
            vxorps(zmmTmp, zmmTmp, zmmTmp);
            vmaxps(zmmC, zmmC, zmmTmp);
            L(L_no_act);
        }
        Label L_full, L_store_done, L_aligned_nt, L_after_nt;
        mov(eax, dword[reg_args + offsetof(gemmv_avx512_fp32_call_args, M_tail)]);
        test(eax, eax);
        jz(L_full);
        // Compute mask k1 = (1 << M_tail) - 1
        mov(ecx, eax);
        mov(eax, 1);
        shl(eax, cl);
        sub(eax, 1);
        kmovw(k1, eax);
        vmovups(ptr[reg_y] | k1, zmmC);
        jmp(L_store_done);
        L(L_full);
        // If y is 64B-aligned, prefer non-temporal store (stream) to reduce cache pollution
        mov(rax, reg_y);
        and_(rax, 63);
        test(rax, rax);
        jnz(L_after_nt);
        vmovntps(ptr[reg_y], zmmC);
        jmp(L_store_done);
        L(L_after_nt);
        vmovups(ptr[reg_y], zmmC);
        L(L_store_done);
    }

    // Epilogue: restore callee-saved regs
    pop(r15);
    pop(r14);
    pop(r13);
    pop(r12);

    ret();
}

JitGemmvAvx512Fp32::JitGemmvAvx512Fp32() {
    fn_ = jit_prebuilt_pool::get_typed<fn_t>(kernel_kind::avx512_fp32);
    OPENVINO_ASSERT(fn_ != nullptr, "jit_gemmv_avx512_fp32 kernel pointer is null");
}

void JitGemmvAvx512Fp32::operator()(const gemmv_ukr_params_t* p) const {
    // Orchestrate over M in blocks of 16; pack layout advances by 16 bytes per K-step.
    gemmv_avx512_fp32_call_args a{};
    a.K = p->K;
    a.gran = static_cast<int>(p->gran);
    a.w_nbits = (p->w_type == w_dtype_t::i4 || p->w_type == w_dtype_t::u4) ? 4 : 8;
    a.w_unsigned = (p->w_type == w_dtype_t::u8 || p->w_type == w_dtype_t::u4) ? 1 : 0;
    a.k_step_bytes = (a.w_nbits == 8 ? 16 : 8);
    a.dbg_enable = p->dbg_enable;
    a.dbg_k = p->dbg_k;
    a.accumulate = p->accumulate ? 1 : 0;
    a.ld_w_bytes = p->ld_w_bytes;
    a.fuse_gate = p->fuse_gate ? 1 : 0;
    a.gate = p->gate_scale;
    a.act_kind = p->act_kind;

    // Decide VNNI path inside this JIT only if explicitly enabled via env
    Xbyak::util::Cpu cpu;
    bool can_vnni = false;
    if (const char* ev = std::getenv("GEMMV_ENABLE_JIT_VNNI")) {
        if (std::string(ev) == "1") {
            can_vnni = cpu.has(Xbyak::util::Cpu::tAVX512_VNNI) && (p->w_type == w_dtype_t::i8) && (a.w_nbits == 8);
        }
    }

    // Quantize X to u8 per-tensor if VNNI path is enabled
    std::vector<uint8_t> xq_buf;
    float s_x = 1.f; int32_t zp_x = 128; int32_t sum_x_q = 0;
    if (can_vnni) {
        const float* xf = static_cast<const float*>(p->x);
        xq_buf.resize(p->K);
        // symmetric per-tensor quant (zp=128)
        float amax = 0.f;
        for (int k = 0; k < p->K; ++k) amax = std::max(amax, std::fabs(xf[k]));
        s_x = (amax > 0.f) ? (amax / 127.f) : 1.f;
        zp_x = 128;
        for (int k = 0; k < p->K; ++k) {
            int v = (int)std::lrintf(xf[k] / s_x) + zp_x;
            if (v < 0) v = 0; if (v > 255) v = 255; xq_buf[k] = (uint8_t)v;
        }
        for (int k = 0; k < p->K; ++k) sum_x_q += (int)xq_buf[k];
    }
    // Precompute sum_x for fp32 zp compensation path
    float sumx = 0.f;
    if (!can_vnni && p->zps) {
        const float* xf = static_cast<const float*>(p->x);
        for (int k = 0; k < p->K; ++k) sumx += xf[k];
    }
    a.sum_x = sumx;

    const int M_blk = 16;
    const int M_full_blocks = p->M / M_blk;
    const int M_tail = p->M % M_blk;

    const float* x = static_cast<const float*>(p->x);
    auto* y = static_cast<float*>(p->y);

    // Iterate full blocks
    for (int bi = 0; bi < M_full_blocks; ++bi) {
        const uint8_t* wblk = p->wq + bi * p->ld_w_bytes;
        float* yblk = y + bi * M_blk;
        // Prepare per-lane metadata pointers
        alignas(64) float sc_buf[M_blk];
        alignas(64) int32_t zp_buf[M_blk];
        alignas(64) float b_buf[M_blk];
        const float* sc_ptr = nullptr;
        const int32_t* zp_ptr = nullptr;
        const float* b_ptr = nullptr;
        if (p->gran == quant_granularity_t::per_tensor) {
            sc_ptr = p->scales; zp_ptr = p->zps; b_ptr = p->bias;
        } else if (p->gran == quant_granularity_t::per_channel) {
            sc_ptr = p->scales + (p->m_base + bi * M_blk);
            zp_ptr = p->zps ? (p->zps + (p->m_base + bi * M_blk)) : nullptr;
            b_ptr  = p->bias ? (p->bias + (p->m_base + bi * M_blk)) : nullptr;
        } else {
            const int base_m = p->m_base + bi * M_blk;
            const int gs = p->group_size > 0 ? p->group_size : M_blk;
            for (int m = 0; m < M_blk; ++m) {
                const int idx_m = base_m + m;
                const int g = idx_m / gs;
                sc_buf[m] = p->scales[g];
                if (p->zps) zp_buf[m] = p->zps[g];
                if (p->bias) b_buf[m] = p->bias[g];
            }
            sc_ptr = sc_buf; zp_ptr = p->zps ? zp_buf : nullptr; b_ptr = p->bias ? b_buf : nullptr;
        }
        if (can_vnni) {
            vnni_norepack_block_u8s8_to_fp32(xq_buf.data(), p->K, wblk, a.k_step_bytes,
                                             sc_ptr, zp_ptr, b_ptr, yblk, M_blk,
                                             p->gran, s_x, zp_x, sum_x_q);
        } else {
            a.x = x; a.wq = wblk; a.scales = sc_ptr; a.zps = zp_ptr; a.bias = b_ptr;
            a.y = yblk; a.M_tail = 0; a.dbg_q = p->dbg_q; a.dbg_qs = p->dbg_qs;
            a.use_vnni = 0; a.need_sumw = 0;
            fn_(&a);
        }
    }

    if (M_tail) {
        const int bi = M_full_blocks;
        const uint8_t* wblk = p->wq + bi * p->ld_w_bytes;
        float* yblk = y + bi * M_blk;
        alignas(64) float sc_buf[M_blk];
        alignas(64) int32_t zp_buf[M_blk];
        alignas(64) float b_buf[M_blk];
        const float* sc_ptr = nullptr;
        const int32_t* zp_ptr = nullptr;
        const float* b_ptr = nullptr;
        if (p->gran == quant_granularity_t::per_tensor) {
            sc_ptr = p->scales; zp_ptr = p->zps; b_ptr = p->bias;
        } else if (p->gran == quant_granularity_t::per_channel) {
            sc_ptr = p->scales + (p->m_base + bi * M_blk);
            zp_ptr = p->zps ? (p->zps + (p->m_base + bi * M_blk)) : nullptr;
            b_ptr  = p->bias ? (p->bias + (p->m_base + bi * M_blk)) : nullptr;
        } else {
            const int base_m = p->m_base + bi * M_blk;
            const int gs = p->group_size > 0 ? p->group_size : M_blk;
            for (int m = 0; m < M_blk; ++m) {
                const int idx_m = base_m + m;
                const int g = idx_m / gs;
                sc_buf[m] = p->scales[g];
                if (p->zps) zp_buf[m] = p->zps[g];
                if (p->bias) b_buf[m] = p->bias[g];
            }
            sc_ptr = sc_buf; zp_ptr = p->zps ? zp_buf : nullptr; b_ptr = p->bias ? b_buf : nullptr;
        }
        if (can_vnni) {
            vnni_norepack_block_u8s8_to_fp32(xq_buf.data(), p->K, wblk, a.k_step_bytes,
                                             sc_ptr, zp_ptr, b_ptr, yblk, M_tail,
                                             p->gran, s_x, zp_x, sum_x_q);
        } else {
            a.x = x; a.wq = wblk; a.scales = sc_ptr; a.zps = zp_ptr; a.bias = b_ptr;
            a.y = yblk; a.M_tail = M_tail; a.dbg_q = p->dbg_q; a.dbg_qs = p->dbg_qs;
            a.use_vnni = 0; a.need_sumw = 0;
            fn_(&a);
        }
    }
}

} // namespace ov::intel_cpu::x64::gemmv_jit
