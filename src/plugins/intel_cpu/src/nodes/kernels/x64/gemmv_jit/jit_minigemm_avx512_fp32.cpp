// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "jit_minigemm_avx512_fp32.hpp"

#include "jit_prebuilt_pool.hpp"
#include "openvino/core/except.hpp"
#include "xbyak/xbyak_util.h"
#include <algorithm>

namespace ov::intel_cpu::x64::gemmv_jit {

static inline int to_int(quant_granularity_t q) { return static_cast<int>(q); }

JitMiniGemmAvx512Fp32::JitMiniGemmAvx512Fp32()
    : dnnl::impl::cpu::x64::jit_generator_t("jit_minigemm_avx512_fp32",
            dnnl::impl::cpu::x64::cpu_isa_t::avx512_core) {
    auto st = create_kernel();
    if (st != dnnl::impl::status::success) {
        OPENVINO_THROW("Failed to build jit_minigemm_avx512_fp32 kernel");
    }
    fn_ = reinterpret_cast<fn_t>(jit_ker());
}

void JitMiniGemmAvx512Fp32::generate() {
    using namespace Xbyak;
    setDefaultJmpNEAR(true);
#if defined(OPENVINO_ARCH_X86_64)
    endbr64();
#endif

    // rdi holds CallArgs*
    const Reg64 reg_args = rdi;
    const Reg64 reg_w = r8;
    const Reg64 reg_sc = r9;
    const Reg64 reg_zp = r10;
    const Reg64 reg_bias = r11;
    const Reg64 reg_k = r12;
    const Reg64 reg_k_iter = r13;
    const Reg64 reg_ncols = r14;

    // Preserve
    push(r12); push(r13); push(r14); push(r15); push(rbx);

    // Load args
    mov(reg_w,    ptr[reg_args + offsetof(CallArgs, wq)]);
    mov(reg_sc,   ptr[reg_args + offsetof(CallArgs, scales)]);
    mov(reg_zp,   ptr[reg_args + offsetof(CallArgs, zps)]);
    mov(reg_bias, ptr[reg_args + offsetof(CallArgs, bias)]);
    mov(eax,      dword[reg_args + offsetof(CallArgs, K)]); mov(reg_k, rax);
    mov(eax,      dword[reg_args + offsetof(CallArgs, n_cols)]); mov(reg_ncols, rax);

    // Prepare X column base pointers in GP regs (rbx, rsi, rdx, r15)
    mov(rbx, ptr[reg_args + offsetof(CallArgs, x0)]);
    mov(rsi, ptr[reg_args + offsetof(CallArgs, x1)]);
    mov(rdx, ptr[reg_args + offsetof(CallArgs, x2)]);
    mov(r15, ptr[reg_args + offsetof(CallArgs, x3)]);

    // Zmms
    const Zmm zC0 = zmm0, zC1 = zmm1, zC2 = zmm2, zC3 = zmm3;
    const Zmm zW  = zmm4;
    const Zmm zS  = zmm5;
    const Zmm zTmp= zmm6;
    const Zmm zX0 = zmm7, zX1 = zmm8, zX2 = zmm9, zX3 = zmm10;
    const Zmm zZP = zmm11, zSZP = zmm12;
    const Xmm xWb = xmm13; // 16 bytes per k (8B for 4-bit path)
    const Ymm yI0 = ymm14, yI1 = ymm15;
    const Ymm yMask000F = ymm12; // word mask 0x000F for 4-bit decode

    // Prepare 0x000F mask for INT4 decode
    {
        Label L_after_mask;
        mov(eax, 0x0F0F0F0F);
        movd(xmm0, eax);
        pshufd(xmm0, xmm0, 0);
        vpmovzxbw(yMask000F, xmm0);
        L(L_after_mask);
    }

    // Load scales into zS (per_tensor vs per_channel)
    {
        Label L_not_pt, L_loaded;
        mov(eax, dword[reg_args + offsetof(CallArgs, gran)]);
        cmp(eax, to_int(quant_granularity_t::per_tensor));
        jne(L_not_pt);
        vbroadcastss(zS, dword[reg_sc]);
        jmp(L_loaded);
        L(L_not_pt);
        {
            Label L_full, L_done;
            mov(eax, dword[reg_args + offsetof(CallArgs, M_tail)]);
            test(eax, eax); jz(L_full);
            mov(ecx, eax); mov(eax, 1); shl(eax, cl); sub(eax, 1); kmovw(k1, eax);
            vmovups(zS | k1 | T_z, ptr[reg_sc]);
            jmp(L_done);
            L(L_full);
            vmovups(zS, ptr[reg_sc]);
            L(L_done);
        }
        L(L_loaded);
    }

    // Init accumulators with 0
    vxorps(zC0, zC0, zC0); vxorps(zC1, zC1, zC1); vxorps(zC2, zC2, zC2); vxorps(zC3, zC3, zC3);

    // K loop
    xor_(reg_k_iter, reg_k_iter);
    // rolling W pointer in rax
    mov(rax, reg_w);
    Xbyak::Label loop_label, done_label;
    L(loop_label);
    cmp(reg_k_iter, reg_k); jge(done_label);

    // Load Wq bytes (16 for 8-bit, 8 for 4-bit), convert to fp32 in zW
    {
        Label L_bits8, L_after_decode;
        mov(eax, dword[reg_args + offsetof(CallArgs, w_nbits)]);
        cmp(eax, 8);
        je(L_bits8);
        // --- 4-bit path ---
        vmovq(xWb, ptr[rax]);
        add(rax, dword[reg_args + offsetof(CallArgs, k_step_bytes)]);
        // widen to words, split nibbles, interleave into 16 bytes
        Xbyak::Xmm xA = Xbyak::Xmm(8);
        Xbyak::Xmm xB = Xbyak::Xmm(9);
        Xbyak::Xmm xZ = Xbyak::Xmm(10);
        vpxor(xZ, xZ, xZ);
        vpmovzxbw(xA, xWb);
        vextracti128(xB, yMask000F, 0);
        Xbyak::Xmm xLow = Xbyak::Xmm(11);
        vmovdqa(xLow, xA); vpand(xLow, xLow, xB);
        Xbyak::Xmm xLo8 = Xbyak::Xmm(12); vpackuswb(xLo8, xLow, xZ);
        Xbyak::Xmm xHigh = Xbyak::Xmm(15);
        vmovdqa(xHigh, xA); vpsrlw(xHigh, xHigh, 4); vpand(xHigh, xHigh, xB);
        Xbyak::Xmm xHi8 = Xbyak::Xmm(1); vpackuswb(xHi8, xHigh, xZ);
        vmovdqa(xWb, xLo8); punpcklbw(xWb, xHi8);
        // bytes -> dword lanes
        vpmovzxbd(yI0, xWb); vpsrldq(xWb, xWb, 8); vpmovzxbd(yI1, xWb);
        {
            Label L_u4, L_after_adj4;
            mov(eax, dword[reg_args + offsetof(CallArgs, w_unsigned)]);
            test(eax, eax); jnz(L_u4);
            // i4: (q ^ 8) - 8 per dword
            Xbyak::Ymm ymmC8 = ymm0;
            mov(eax, 8); movd(xZ, eax); pshufd(xZ, xZ, 0); vpbroadcastd(ymmC8, xZ);
            vpxor(yI0, yI0, ymmC8); vpsubd(yI0, yI0, ymmC8);
            vpxor(yI1, yI1, ymmC8); vpsubd(yI1, yI1, ymmC8);
            L(L_u4);
            L(L_after_adj4);
        }
        vcvtdq2ps(ymm0, yI0); vcvtdq2ps(ymm1, yI1);
        vxorps(zW, zW, zW); vinsertf32x8(zW, zW, ymm0, 0); vinsertf32x8(zW, zW, ymm1, 1);
        jmp(L_after_decode);
        // --- 8-bit path ---
        L(L_bits8);
        // Prefetch next lines to hide some L1 miss (software prefetch, cheap & safe)
        prefetcht0(ptr[rax + 64]);
        movdqu(xWb, ptr[rax]);
        add(rax, dword[reg_args + offsetof(CallArgs, k_step_bytes)]);
        {
            Label L_u8, L_after8;
            mov(eax, dword[reg_args + offsetof(CallArgs, is_u8)]);
            test(eax, eax); jz(L_u8);
            vpmovzxbd(yI0, xWb); vcvtdq2ps(ymm0, yI0);
            vpsrldq(xWb, xWb, 8); vpmovzxbd(yI1, xWb); vcvtdq2ps(ymm1, yI1);
            vxorps(zW, zW, zW); vinsertf32x8(zW, zW, ymm0, 0); vinsertf32x8(zW, zW, ymm1, 1);
            jmp(L_after8);
            L(L_u8);
            vpmovsxbd(yI0, xWb); vcvtdq2ps(ymm0, yI0);
            vpsrldq(xWb, xWb, 8); vpmovsxbd(yI1, xWb); vcvtdq2ps(ymm1, yI1);
            vxorps(zW, zW, zW); vinsertf32x8(zW, zW, ymm0, 0); vinsertf32x8(zW, zW, ymm1, 1);
            L(L_after8);
        }
        L(L_after_decode);
    }
    // dequant: w *= s
    vmulps(zW, zW, zS);

    // Broadcast X scalars for up to 4 columns, then FMA
    {
        // x0
        vbroadcastss(zX0, dword[rbx + reg_k_iter * 4]);
        vfmadd231ps(zC0, zW, zX0);
        // x1 if n_cols >= 2
        Label L_c1, L_c2, L_c3, L_after_cols;
        cmp(reg_ncols, 2); jl(L_c1);
        vbroadcastss(zX1, dword[rsi + reg_k_iter * 4]);
        vfmadd231ps(zC1, zW, zX1);
        L(L_c1);
        cmp(reg_ncols, 3); jl(L_c2);
        vbroadcastss(zX2, dword[rdx + reg_k_iter * 4]);
        vfmadd231ps(zC2, zW, zX2);
        L(L_c2);
        cmp(reg_ncols, 4); jl(L_c3);
        vbroadcastss(zX3, dword[r15 + reg_k_iter * 4]);
        vfmadd231ps(zC3, zW, zX3);
        L(L_c3);
        L(L_after_cols);
    }

    inc(reg_k_iter);
    // Try second unrolled step if k_iter < K
    {
        Label L_do_second, L_after_second;
        cmp(reg_k_iter, reg_k); jl(L_do_second);
        jmp(L_after_second);
        L(L_do_second);

        // --- second unrolled step (unique labels) ---
        {
            Label L_bits8b, L_after_decodeb;
            mov(eax, dword[reg_args + offsetof(CallArgs, w_nbits)]);
            cmp(eax, 8);
            je(L_bits8b);
            // 4-bit path
            vmovq(xWb, ptr[rax]);
            add(rax, dword[reg_args + offsetof(CallArgs, k_step_bytes)]);
            Xbyak::Xmm xA = Xbyak::Xmm(8);
            Xbyak::Xmm xB = Xbyak::Xmm(11);
            Xbyak::Xmm xZ = Xbyak::Xmm(10);
            vpxor(xZ, xZ, xZ);
            vpmovzxbw(xA, xWb);
            vextracti128(xB, yMask000F, 0);
            Xbyak::Xmm xLow = Xbyak::Xmm(11);
            vmovdqa(xLow, xA); vpand(xLow, xLow, xB);
            Xbyak::Xmm xLo8 = Xbyak::Xmm(12); vpackuswb(xLo8, xLow, xZ);
            Xbyak::Xmm xHigh = Xbyak::Xmm(15);
            vmovdqa(xHigh, xA); vpsrlw(xHigh, xHigh, 4); vpand(xHigh, xHigh, xB);
            Xbyak::Xmm xHi8 = Xbyak::Xmm(1); vpackuswb(xHi8, xHigh, xZ);
            vmovdqa(xWb, xLo8); punpcklbw(xWb, xHi8);
            vpmovzxbd(yI0, xWb); vpsrldq(xWb, xWb, 8); vpmovzxbd(yI1, xWb);
            {
                Label L_u4b, L_after_adj4b;
                mov(eax, dword[reg_args + offsetof(CallArgs, w_unsigned)]);
                test(eax, eax); jnz(L_u4b);
                Xbyak::Ymm ymmC8b = ymm0;
                mov(eax, 8); movd(xZ, eax); pshufd(xZ, xZ, 0); vpbroadcastd(ymmC8b, xZ);
                vpxor(yI0, yI0, ymmC8b); vpsubd(yI0, yI0, ymmC8b);
                vpxor(yI1, yI1, ymmC8b); vpsubd(yI1, yI1, ymmC8b);
                L(L_u4b);
                L(L_after_adj4b);
            }
            vcvtdq2ps(ymm0, yI0); vcvtdq2ps(ymm1, yI1);
            vxorps(zW, zW, zW); vinsertf32x8(zW, zW, ymm0, 0); vinsertf32x8(zW, zW, ymm1, 1);
            jmp(L_after_decodeb);
            // 8-bit path
            L(L_bits8b);
            prefetcht0(ptr[rax + 64]);
            movdqu(xWb, ptr[rax]);
            add(rax, dword[reg_args + offsetof(CallArgs, k_step_bytes)]);
            {
                Label L_u8b, L_after8b;
                mov(eax, dword[reg_args + offsetof(CallArgs, is_u8)]);
                test(eax, eax); jz(L_u8b);
                vpmovzxbd(yI0, xWb); vcvtdq2ps(ymm0, yI0);
                vpsrldq(xWb, xWb, 8); vpmovzxbd(yI1, xWb); vcvtdq2ps(ymm1, yI1);
                vxorps(zW, zW, zW); vinsertf32x8(zW, zW, ymm0, 0); vinsertf32x8(zW, zW, ymm1, 1);
                jmp(L_after8b);
                L(L_u8b);
                vpmovsxbd(yI0, xWb); vcvtdq2ps(ymm0, yI0);
                vpsrldq(xWb, xWb, 8); vpmovsxbd(yI1, xWb); vcvtdq2ps(ymm1, yI1);
                vxorps(zW, zW, zW); vinsertf32x8(zW, zW, ymm0, 0); vinsertf32x8(zW, zW, ymm1, 1);
                L(L_after8b);
            }
            L(L_after_decodeb);
        }
        vmulps(zW, zW, zS);
        {
            // x broadcasts for 2nd step
            vbroadcastss(zX0, dword[rbx + reg_k_iter * 4]);
            vfmadd231ps(zC0, zW, zX0);
            Label L_c1b, L_c2b, L_c3b;
            cmp(reg_ncols, 2); jl(L_c1b);
            vbroadcastss(zX1, dword[rsi + reg_k_iter * 4]);
            vfmadd231ps(zC1, zW, zX1);
            L(L_c1b);
            cmp(reg_ncols, 3); jl(L_c2b);
            vbroadcastss(zX2, dword[rdx + reg_k_iter * 4]);
            vfmadd231ps(zC2, zW, zX2);
            L(L_c2b);
            cmp(reg_ncols, 4); jl(L_c3b);
            vbroadcastss(zX3, dword[r15 + reg_k_iter * 4]);
            vfmadd231ps(zC3, zW, zX3);
            L(L_c3b);
        }
        inc(reg_k_iter);
        // --- third unrolled step if available ---
        {
            Label L_do_third, L_bits8c, L_after_decodec, L_u4c, L_after_adj4c, L_u8c, L_after8c;
            cmp(reg_k_iter, reg_k); jl(L_do_third);
            jmp(L_after_second);
            L(L_do_third);
            mov(eax, dword[reg_args + offsetof(CallArgs, w_nbits)]);
            cmp(eax, 8);
            je(L_bits8c);
            vmovq(xWb, ptr[rax]);
            add(rax, dword[reg_args + offsetof(CallArgs, k_step_bytes)]);
            Xbyak::Xmm xA = Xbyak::Xmm(8);
            Xbyak::Xmm xB = Xbyak::Xmm(11);
            Xbyak::Xmm xZ = Xbyak::Xmm(10);
            vpxor(xZ, xZ, xZ);
            vpmovzxbw(xA, xWb);
            vextracti128(xB, yMask000F, 0);
            Xbyak::Xmm xLow = Xbyak::Xmm(11);
            vmovdqa(xLow, xA); vpand(xLow, xLow, xB);
            Xbyak::Xmm xLo8 = Xbyak::Xmm(12); vpackuswb(xLo8, xLow, xZ);
            Xbyak::Xmm xHigh = Xbyak::Xmm(15);
            vmovdqa(xHigh, xA); vpsrlw(xHigh, xHigh, 4); vpand(xHigh, xHigh, xB);
            Xbyak::Xmm xHi8 = Xbyak::Xmm(1); vpackuswb(xHi8, xHigh, xZ);
            vmovdqa(xWb, xLo8); punpcklbw(xWb, xHi8);
            vpmovzxbd(yI0, xWb); vpsrldq(xWb, xWb, 8); vpmovzxbd(yI1, xWb);
            mov(eax, dword[reg_args + offsetof(CallArgs, w_unsigned)]);
            test(eax, eax); jnz(L_u4c);
            {
                Xbyak::Ymm ymmC8c = ymm0;
                mov(eax, 8); movd(xZ, eax); pshufd(xZ, xZ, 0); vpbroadcastd(ymmC8c, xZ);
                vpxor(yI0, yI0, ymmC8c); vpsubd(yI0, yI0, ymmC8c);
                vpxor(yI1, yI1, ymmC8c); vpsubd(yI1, yI1, ymmC8c);
            }
            L(L_u4c);
            vcvtdq2ps(ymm0, yI0); vcvtdq2ps(ymm1, yI1);
            vxorps(zW, zW, zW); vinsertf32x8(zW, zW, ymm0, 0); vinsertf32x8(zW, zW, ymm1, 1);
            jmp(L_after_decodec);
            L(L_bits8c);
            prefetcht0(ptr[rax + 64]);
            movdqu(xWb, ptr[rax]);
            add(rax, dword[reg_args + offsetof(CallArgs, k_step_bytes)]);
            mov(eax, dword[reg_args + offsetof(CallArgs, is_u8)]);
            test(eax, eax); jz(L_u8c);
            vpmovzxbd(yI0, xWb); vcvtdq2ps(ymm0, yI0);
            vpsrldq(xWb, xWb, 8); vpmovzxbd(yI1, xWb); vcvtdq2ps(ymm1, yI1);
            vxorps(zW, zW, zW); vinsertf32x8(zW, zW, ymm0, 0); vinsertf32x8(zW, zW, ymm1, 1);
            jmp(L_after8c);
            L(L_u8c);
            vpmovsxbd(yI0, xWb); vcvtdq2ps(ymm0, yI0);
            vpsrldq(xWb, xWb, 8); vpmovsxbd(yI1, xWb); vcvtdq2ps(ymm1, yI1);
            vxorps(zW, zW, zW); vinsertf32x8(zW, zW, ymm0, 0); vinsertf32x8(zW, zW, ymm1, 1);
            L(L_after8c);
            L(L_after_decodec);
            vmulps(zW, zW, zS);
            // FMA with X broadcasts
            vbroadcastss(zX0, dword[rbx + reg_k_iter * 4]); vfmadd231ps(zC0, zW, zX0);
            {
                Label L_c1c, L_c2c, L_c3c;
                cmp(reg_ncols, 2); jl(L_c1c); vbroadcastss(zX1, dword[rsi + reg_k_iter * 4]); vfmadd231ps(zC1, zW, zX1); L(L_c1c);
                cmp(reg_ncols, 3); jl(L_c2c); vbroadcastss(zX2, dword[rdx + reg_k_iter * 4]); vfmadd231ps(zC2, zW, zX2); L(L_c2c);
                cmp(reg_ncols, 4); jl(L_c3c); vbroadcastss(zX3, dword[r15 + reg_k_iter * 4]); vfmadd231ps(zC3, zW, zX3); L(L_c3c);
            }
            inc(reg_k_iter);
        }

        // --- fourth unrolled step if available ---
        {
            Label L_do_fourth, L_bits8d, L_after_decoded, L_u4d, L_after_adj4d, L_u8d, L_after8d;
            cmp(reg_k_iter, reg_k); jl(L_do_fourth);
            jmp(L_after_second);
            L(L_do_fourth);
            mov(eax, dword[reg_args + offsetof(CallArgs, w_nbits)]);
            cmp(eax, 8);
            je(L_bits8d);
            vmovq(xWb, ptr[rax]);
            add(rax, dword[reg_args + offsetof(CallArgs, k_step_bytes)]);
            Xbyak::Xmm xA = Xbyak::Xmm(8);
            Xbyak::Xmm xB = Xbyak::Xmm(11);
            Xbyak::Xmm xZ = Xbyak::Xmm(10);
            vpxor(xZ, xZ, xZ);
            vpmovzxbw(xA, xWb);
            vextracti128(xB, yMask000F, 0);
            Xbyak::Xmm xLow = Xbyak::Xmm(11);
            vmovdqa(xLow, xA); vpand(xLow, xLow, xB);
            Xbyak::Xmm xLo8 = Xbyak::Xmm(12); vpackuswb(xLo8, xLow, xZ);
            Xbyak::Xmm xHigh = Xbyak::Xmm(15);
            vmovdqa(xHigh, xA); vpsrlw(xHigh, xHigh, 4); vpand(xHigh, xHigh, xB);
            Xbyak::Xmm xHi8 = Xbyak::Xmm(1); vpackuswb(xHi8, xHigh, xZ);
            vmovdqa(xWb, xLo8); punpcklbw(xWb, xHi8);
            vpmovzxbd(yI0, xWb); vpsrldq(xWb, xWb, 8); vpmovzxbd(yI1, xWb);
            mov(eax, dword[reg_args + offsetof(CallArgs, w_unsigned)]);
            test(eax, eax); jnz(L_u4d);
            {
                Xbyak::Ymm ymmC8d = ymm0;
                mov(eax, 8); movd(xZ, eax); pshufd(xZ, xZ, 0); vpbroadcastd(ymmC8d, xZ);
                vpxor(yI0, yI0, ymmC8d); vpsubd(yI0, yI0, ymmC8d);
                vpxor(yI1, yI1, ymmC8d); vpsubd(yI1, yI1, ymmC8d);
            }
            L(L_u4d);
            vcvtdq2ps(ymm0, yI0); vcvtdq2ps(ymm1, yI1);
            vxorps(zW, zW, zW); vinsertf32x8(zW, zW, ymm0, 0); vinsertf32x8(zW, zW, ymm1, 1);
            jmp(L_after_decoded);
            L(L_bits8d);
            prefetcht0(ptr[rax + 64]);
            movdqu(xWb, ptr[rax]);
            add(rax, dword[reg_args + offsetof(CallArgs, k_step_bytes)]);
            mov(eax, dword[reg_args + offsetof(CallArgs, is_u8)]);
            test(eax, eax); jz(L_u8d);
            vpmovzxbd(yI0, xWb); vcvtdq2ps(ymm0, yI0);
            vpsrldq(xWb, xWb, 8); vpmovzxbd(yI1, xWb); vcvtdq2ps(ymm1, yI1);
            vxorps(zW, zW, zW); vinsertf32x8(zW, zW, ymm0, 0); vinsertf32x8(zW, zW, ymm1, 1);
            jmp(L_after8d);
            L(L_u8d);
            vpmovsxbd(yI0, xWb); vcvtdq2ps(ymm0, yI0);
            vpsrldq(xWb, xWb, 8); vpmovsxbd(yI1, xWb); vcvtdq2ps(ymm1, yI1);
            vxorps(zW, zW, zW); vinsertf32x8(zW, zW, ymm0, 0); vinsertf32x8(zW, zW, ymm1, 1);
            L(L_after8d);
            L(L_after_decoded);
            vmulps(zW, zW, zS);
            // FMA with X broadcasts
            vbroadcastss(zX0, dword[rbx + reg_k_iter * 4]); vfmadd231ps(zC0, zW, zX0);
            {
                Label L_c1d, L_c2d, L_c3d;
                cmp(reg_ncols, 2); jl(L_c1d); vbroadcastss(zX1, dword[rsi + reg_k_iter * 4]); vfmadd231ps(zC1, zW, zX1); L(L_c1d);
                cmp(reg_ncols, 3); jl(L_c2d); vbroadcastss(zX2, dword[rdx + reg_k_iter * 4]); vfmadd231ps(zC2, zW, zX2); L(L_c2d);
                cmp(reg_ncols, 4); jl(L_c3d); vbroadcastss(zX3, dword[r15 + reg_k_iter * 4]); vfmadd231ps(zC3, zW, zX3); L(L_c3d);
            }
            inc(reg_k_iter);
        }
        L(L_after_second);
    }

        jmp(loop_label);
    L(done_label);

    // Bias add + ZP compensation per column
    // Load bias (per_tensor/per_channel) to zTmp and add to each C
    {
        Label L_no_bias, L_bias_pt, L_done_bias;
        mov(rax, reg_bias); test(rax, rax); jz(L_no_bias);
        mov(eax, dword[reg_args + offsetof(CallArgs, gran)]);
        cmp(eax, to_int(quant_granularity_t::per_tensor)); jne(L_bias_pt);
        vbroadcastss(zTmp, dword[reg_bias]);
        vaddps(zC0, zC0, zTmp);
        cmp(reg_ncols, 2); jl(L_done_bias); vaddps(zC1, zC1, zTmp);
        cmp(reg_ncols, 3); jl(L_done_bias); vaddps(zC2, zC2, zTmp);
        cmp(reg_ncols, 4); jl(L_done_bias); vaddps(zC3, zC3, zTmp);
        jmp(L_done_bias);
        L(L_bias_pt);
        {
            Label L_full, L_done;
            mov(eax, dword[reg_args + offsetof(CallArgs, M_tail)]);
            test(eax, eax); jz(L_full);
            mov(ecx, eax); mov(eax, 1); shl(eax, cl); sub(eax, 1); kmovw(k1, eax);
            vmovups(zTmp | k1 | T_z, ptr[reg_bias]);
            jmp(L_done);
            L(L_full);
            vmovups(zTmp, ptr[reg_bias]);
            L(L_done);
        }
        vaddps(zC0, zC0, zTmp);
        cmp(reg_ncols, 2); jl(L_done_bias); vaddps(zC1, zC1, zTmp);
        cmp(reg_ncols, 3); jl(L_done_bias); vaddps(zC2, zC2, zTmp);
        cmp(reg_ncols, 4); jl(L_done_bias); vaddps(zC3, zC3, zTmp);
        L(L_done_bias);
        L(L_no_bias);
    }

    // ZP compensation: Cn -= s * zp * sum_xn
    {
        Label L_no_zp, L_pt, L_done_zp;
        mov(rax, reg_zp); test(rax, rax); jz(L_no_zp);
        mov(eax, dword[reg_args + offsetof(CallArgs, gran)]);
        cmp(eax, to_int(quant_granularity_t::per_tensor)); jne(L_pt);
        // per_tensor: broadcast zp, compute s*zp once in zSZP
        mov(eax, dword[reg_zp]);
        vcvtsi2ss(xmm0, xmm0, eax); vbroadcastss(zZP, xmm0);
        vmulps(zSZP, zS, zZP);
        // subtract per column
        vbroadcastss(zX0, dword[reg_args + offsetof(CallArgs, sum_x0)]);
        vmulps(zTmp, zSZP, zX0); vsubps(zC0, zC0, zTmp);
        cmp(reg_ncols, 2); jl(L_done_zp);
        vbroadcastss(zX1, dword[reg_args + offsetof(CallArgs, sum_x1)]);
        vmulps(zTmp, zSZP, zX1); vsubps(zC1, zC1, zTmp);
        cmp(reg_ncols, 3); jl(L_done_zp);
        vbroadcastss(zX2, dword[reg_args + offsetof(CallArgs, sum_x2)]);
        vmulps(zTmp, zSZP, zX2); vsubps(zC2, zC2, zTmp);
        cmp(reg_ncols, 4); jl(L_done_zp);
        vbroadcastss(zX3, dword[reg_args + offsetof(CallArgs, sum_x3)]);
        vmulps(zTmp, zSZP, zX3); vsubps(zC3, zC3, zTmp);
        jmp(L_done_zp);
        L(L_pt);
        // per_channel: load zp vector (masked if tail), compute s*zp vector, then scale by sum_xn
        {
            Label L_full, L_done;
            mov(eax, dword[reg_args + offsetof(CallArgs, M_tail)]);
            test(eax, eax); jz(L_full);
            mov(ecx, eax); mov(eax, 1); shl(eax, cl); sub(eax, 1); kmovw(k1, eax);
            vmovdqu32(zZP | k1 | T_z, ptr[reg_zp]);
            jmp(L_done);
            L(L_full);
            vmovdqu32(zZP, ptr[reg_zp]);
            L(L_done);
        }
        vcvtdq2ps(zZP, zZP);
        vmulps(zSZP, zS, zZP);
        // per column
        vbroadcastss(zX0, dword[reg_args + offsetof(CallArgs, sum_x0)]);
        vmulps(zTmp, zSZP, zX0); vsubps(zC0, zC0, zTmp);
        cmp(reg_ncols, 2); jl(L_done_zp);
        vbroadcastss(zX1, dword[reg_args + offsetof(CallArgs, sum_x1)]);
        vmulps(zTmp, zSZP, zX1); vsubps(zC1, zC1, zTmp);
        cmp(reg_ncols, 3); jl(L_done_zp);
        vbroadcastss(zX2, dword[reg_args + offsetof(CallArgs, sum_x2)]);
        vmulps(zTmp, zSZP, zX2); vsubps(zC2, zC2, zTmp);
        cmp(reg_ncols, 4); jl(L_done_zp);
        vbroadcastss(zX3, dword[reg_args + offsetof(CallArgs, sum_x3)]);
        vmulps(zTmp, zSZP, zX3); vsubps(zC3, zC3, zTmp);
        L(L_done_zp);
        L(L_no_zp);
    }

    // Store Y with M_tail mask as needed
    {
        // MoE gate: y[n] *= gate[n]
        vbroadcastss(zX0, dword[reg_args + offsetof(CallArgs, gate0)]);
        vmulps(zC0, zC0, zX0);
        Label L_gate1, L_gate2, L_gate3, L_gate_done;
        cmp(reg_ncols, 2); jl(L_gate1);
        vbroadcastss(zX1, dword[reg_args + offsetof(CallArgs, gate1)]);
        vmulps(zC1, zC1, zX1);
        L(L_gate1);
        cmp(reg_ncols, 3); jl(L_gate2);
        vbroadcastss(zX2, dword[reg_args + offsetof(CallArgs, gate2)]);
        vmulps(zC2, zC2, zX2);
        L(L_gate2);
        cmp(reg_ncols, 4); jl(L_gate3);
        vbroadcastss(zX3, dword[reg_args + offsetof(CallArgs, gate3)]);
        vmulps(zC3, zC3, zX3);
        L(L_gate3);
        L(L_gate_done);

        // Optional activation: act_kind==1 -> ReLU
        {
            Label L_no_act, L_after_act;
            mov(eax, dword[reg_args + offsetof(CallArgs, act_kind)]);
            cmp(eax, 1); jne(L_no_act);
            vxorps(zTmp, zTmp, zTmp);
            vmaxps(zC0, zC0, zTmp);
            cmp(reg_ncols, 2); jl(L_after_act); vmaxps(zC1, zC1, zTmp);
            cmp(reg_ncols, 3); jl(L_after_act); vmaxps(zC2, zC2, zTmp);
            cmp(reg_ncols, 4); jl(L_after_act); vmaxps(zC3, zC3, zTmp);
            L(L_after_act);
            L(L_no_act);
        }

        Label L_full, L_done;
        mov(eax, dword[reg_args + offsetof(CallArgs, M_tail)]);
        test(eax, eax); jz(L_full);
        mov(ecx, eax); mov(eax, 1); shl(eax, cl); sub(eax, 1); kmovw(k1, eax);
        mov(r15, ptr[reg_args + offsetof(CallArgs, y0)]); vmovups(ptr[r15] | k1, zC0);
        cmp(reg_ncols, 2); jl(L_done); mov(r15, ptr[reg_args + offsetof(CallArgs, y1)]); vmovups(ptr[r15] | k1, zC1);
        cmp(reg_ncols, 3); jl(L_done); mov(r15, ptr[reg_args + offsetof(CallArgs, y2)]); vmovups(ptr[r15] | k1, zC2);
        cmp(reg_ncols, 4); jl(L_done); mov(r15, ptr[reg_args + offsetof(CallArgs, y3)]); vmovups(ptr[r15] | k1, zC3);
        jmp(L_done);
        L(L_full);
        mov(r15, ptr[reg_args + offsetof(CallArgs, y0)]); vmovups(ptr[r15], zC0);
        cmp(reg_ncols, 2); jl(L_done); mov(r15, ptr[reg_args + offsetof(CallArgs, y1)]); vmovups(ptr[r15], zC1);
        cmp(reg_ncols, 3); jl(L_done); mov(r15, ptr[reg_args + offsetof(CallArgs, y2)]); vmovups(ptr[r15], zC2);
        cmp(reg_ncols, 4); jl(L_done); mov(r15, ptr[reg_args + offsetof(CallArgs, y3)]); vmovups(ptr[r15], zC3);
        L(L_done);
    }

    // Restore
    pop(rbx); pop(r15); pop(r14); pop(r13); pop(r12);
    ret();
}

bool run_minigemm_jit_i8u8_fp32(const float* x, int K, int N,
                                const uint8_t* wq_packed, int M, int ld_w_bytes,
                                const float* scales, const int32_t* zps,
                                float* y, const float* bias,
                                quant_granularity_t gran, bool is_u8) {
    // Guard ISA
    Xbyak::util::Cpu cpu;
    if (!cpu.has(Xbyak::util::Cpu::tAVX512F)) return false;
    if (gran == quant_granularity_t::per_group) return false; // ref path for now
    if (N <= 0) return true;

    auto fn = jit_prebuilt_pool::get_typed<JitMiniGemmAvx512Fp32::fn_t>(kernel_kind::minigemm_avx512_fp32);
    OPENVINO_ASSERT(fn != nullptr, "mini-GEMM kernel pointer is null");

    const int M_blk = 16;
    const int M_full = M / M_blk;
    const int M_tail = M % M_blk;
    auto get_sc_ptr = [&](int bi){ return (gran == quant_granularity_t::per_tensor) ? scales : (scales + bi * M_blk); };
    auto get_zp_ptr = [&](int bi){ return (!zps) ? nullptr : (gran == quant_granularity_t::per_tensor ? zps : (zps + bi * M_blk)); };
    auto get_bs_ptr = [&](int bi){ return (!bias) ? nullptr : (gran == quant_granularity_t::per_tensor ? bias : (bias + bi * M_blk)); };

    // Precompute per-column sum_x
    float sx[4] = {0,0,0,0};
    for (int n = 0; n < N; ++n) {
        float s=0; for (int k = 0; k < K; ++k) s += x[(size_t)n*K + k]; sx[n] = s;
    }

    auto run_block = [&](int bi, int valid){
        JitMiniGemmAvx512Fp32::CallArgs a{};
        const int base_m = bi * M_blk;
        // Prepare XY pointers for up to 4 columns (column-major Y)
        const int n_cols = std::min(N, 4);
        a.n_cols = n_cols;
        a.x0 = (n_cols >= 1) ? (x + 0*K) : nullptr;
        a.x1 = (n_cols >= 2) ? (x + 1*K) : nullptr;
        a.x2 = (n_cols >= 3) ? (x + 2*K) : nullptr;
        a.x3 = (n_cols >= 4) ? (x + 3*K) : nullptr;
        a.y0 = (n_cols >= 1) ? (y + 0*(size_t)M + base_m) : nullptr;
        a.y1 = (n_cols >= 2) ? (y + 1*(size_t)M + base_m) : nullptr;
        a.y2 = (n_cols >= 3) ? (y + 2*(size_t)M + base_m) : nullptr;
        a.y3 = (n_cols >= 4) ? (y + 3*(size_t)M + base_m) : nullptr;
        a.sum_x0 = sx[0]; a.sum_x1 = sx[1]; a.sum_x2 = sx[2]; a.sum_x3 = sx[3];
        a.wq = wq_packed + (size_t)bi * (size_t)ld_w_bytes;
        a.scales = get_sc_ptr(bi);
        a.zps = get_zp_ptr(bi);
        a.bias = get_bs_ptr(bi);
        a.K = K; a.M_tail = (valid==M_blk?0:valid);
        a.gran = static_cast<int>(gran);
        a.is_u8 = is_u8 ? 1 : 0;
        a.w_nbits = 8;
        a.w_unsigned = is_u8 ? 1 : 0;
        a.k_step_bytes = 16; // 8-bit
        fn(&a);
    };

    // Process columns in tiles of 4
    int n_done = 0;
    while (n_done < N) {
        int tile_max = 4;
        if (K < 512) tile_max = 2;           // маленький K → меньше колонок выгоднее
        if (M < 64) tile_max = std::min(tile_max, 2);
        const int n_tile = std::min(tile_max, N - n_done);
        // Advance x,y pointers for this tile by offsetting base
        const float* xt = x + (size_t)n_done * (size_t)K;
        float* yt = y + (size_t)n_done * (size_t)M;
        // temporarily alias x,y for tile
        // Run full blocks
        for (int bi = 0; bi < M_full; ++bi) {
            // Patch call per tile by passing xt/yt in run_block via static offsets
            // We reuse run_block by adjusting globals (not thread-safe; this function is local scope)
            // Simpler: adjust pointers on the fly
            JitMiniGemmAvx512Fp32::CallArgs a{};
            const int base_m = bi * M_blk;
            a.n_cols = n_tile;
            a.x0 = (n_tile >= 1) ? (xt + 0*K) : nullptr;
            a.x1 = (n_tile >= 2) ? (xt + 1*K) : nullptr;
            a.x2 = (n_tile >= 3) ? (xt + 2*K) : nullptr;
            a.x3 = (n_tile >= 4) ? (xt + 3*K) : nullptr;
            a.y0 = (n_tile >= 1) ? (yt + 0*(size_t)M + base_m) : nullptr;
            a.y1 = (n_tile >= 2) ? (yt + 1*(size_t)M + base_m) : nullptr;
            a.y2 = (n_tile >= 3) ? (yt + 2*(size_t)M + base_m) : nullptr;
            a.y3 = (n_tile >= 4) ? (yt + 3*(size_t)M + base_m) : nullptr;
            // sums for this tile
            float lsx[4] = {0,0,0,0};
            for (int t = 0; t < n_tile; ++t) { float s=0; for (int k=0;k<K;++k) s += xt[(size_t)t*K + k]; lsx[t]=s; }
            a.sum_x0 = lsx[0]; a.sum_x1 = lsx[1]; a.sum_x2 = lsx[2]; a.sum_x3 = lsx[3];
            a.gate0 = 1.f; a.gate1 = 1.f; a.gate2 = 1.f; a.gate3 = 1.f; a.act_kind = 0;
            a.wq = wq_packed + (size_t)bi * (size_t)ld_w_bytes;
            a.scales = get_sc_ptr(bi);
            a.zps = get_zp_ptr(bi);
            a.bias = get_bs_ptr(bi);
            a.K = K; a.M_tail = 0; a.gran = static_cast<int>(gran);
            a.is_u8 = is_u8 ? 1 : 0; a.w_nbits = 8; a.w_unsigned = is_u8 ? 1 : 0; a.k_step_bytes = 16;
            fn(&a);
        }
        if (M_tail) {
            const int bi = M_full;
            JitMiniGemmAvx512Fp32::CallArgs a{};
            const int base_m = bi * M_blk;
            a.n_cols = n_tile;
            a.x0 = (n_tile >= 1) ? (xt + 0*K) : nullptr;
            a.x1 = (n_tile >= 2) ? (xt + 1*K) : nullptr;
            a.x2 = (n_tile >= 3) ? (xt + 2*K) : nullptr;
            a.x3 = (n_tile >= 4) ? (xt + 3*K) : nullptr;
            a.y0 = (n_tile >= 1) ? (yt + 0*(size_t)M + base_m) : nullptr;
            a.y1 = (n_tile >= 2) ? (yt + 1*(size_t)M + base_m) : nullptr;
            a.y2 = (n_tile >= 3) ? (yt + 2*(size_t)M + base_m) : nullptr;
            a.y3 = (n_tile >= 4) ? (yt + 3*(size_t)M + base_m) : nullptr;
            float lsx[4] = {0,0,0,0};
            for (int t = 0; t < n_tile; ++t) { float s=0; for (int k=0;k<K;++k) s += xt[(size_t)t*K + k]; lsx[t]=s; }
            a.sum_x0 = lsx[0]; a.sum_x1 = lsx[1]; a.sum_x2 = lsx[2]; a.sum_x3 = lsx[3];
            a.gate0 = 1.f; a.gate1 = 1.f; a.gate2 = 1.f; a.gate3 = 1.f; a.act_kind = 0;
            a.wq = wq_packed + (size_t)bi * (size_t)ld_w_bytes;
            a.scales = get_sc_ptr(bi);
            a.zps = get_zp_ptr(bi);
            a.bias = get_bs_ptr(bi);
            a.K = K; a.M_tail = M_tail; a.gran = static_cast<int>(gran);
            a.is_u8 = is_u8 ? 1 : 0; a.w_nbits = 8; a.w_unsigned = is_u8 ? 1 : 0; a.k_step_bytes = 16;
            fn(&a);
        }
        n_done += n_tile;
    }
    return true;
}

bool run_minigemm_jit_q_fp32(const float* x, int K, int N,
                             const uint8_t* wq_packed, int M, int ld_w_bytes,
                             const float* scales, const int32_t* zps,
                             float* y, const float* bias,
                             quant_granularity_t gran, w_dtype_t wtype, int group_size) {
    // Guard ISA
    Xbyak::util::Cpu cpu;
    if (!cpu.has(Xbyak::util::Cpu::tAVX512F)) return false;
    if (N <= 0) return true;

    auto fn = jit_prebuilt_pool::get_typed<JitMiniGemmAvx512Fp32::fn_t>(kernel_kind::minigemm_avx512_fp32);
    OPENVINO_ASSERT(fn != nullptr, "mini-GEMM kernel pointer is null");

    const int M_blk = 16;
    const int M_full = M / M_blk;
    const int M_tail = M % M_blk;
    const int w_nbits = (wtype == w_dtype_t::i4 || wtype == w_dtype_t::u4) ? 4 : 8;
    const int w_unsigned = (wtype == w_dtype_t::u8 || wtype == w_dtype_t::u4) ? 1 : 0;
    const int k_step_bytes = (w_nbits == 8 ? 16 : 8);

    auto get_sc_ptr = [&](int bi){ return (gran == quant_granularity_t::per_tensor) ? scales : (scales + bi * M_blk); };
    auto get_zp_ptr = [&](int bi){ return (!zps) ? nullptr : (gran == quant_granularity_t::per_tensor ? zps : (zps + bi * M_blk)); };
    auto get_bs_ptr = [&](int bi){ return (!bias) ? nullptr : (gran == quant_granularity_t::per_tensor ? bias : (bias + bi * M_blk)); };

    // Precompute per-column sum_x
    // Compute on each tile as in i8/u8 path to handle large N

    // Process columns in tiles of 4
    int n_done = 0;
    while (n_done < N) {
        const int n_tile = std::min(4, N - n_done);
        const float* xt = x + (size_t)n_done * (size_t)K;
        float* yt = y + (size_t)n_done * (size_t)M;
        float sx[4] = {0,0,0,0};
        for (int t = 0; t < n_tile; ++t) { float s=0; for (int k = 0; k < K; ++k) s += xt[(size_t)t*K + k]; sx[t]=s; }

        // Full blocks
        for (int bi = 0; bi < M_full; ++bi) {
            JitMiniGemmAvx512Fp32::CallArgs a{};
            const int base_m = bi * M_blk;
            a.n_cols = n_tile;
            a.x0 = (n_tile >= 1) ? (xt + 0*K) : nullptr;
            a.x1 = (n_tile >= 2) ? (xt + 1*K) : nullptr;
            a.x2 = (n_tile >= 3) ? (xt + 2*K) : nullptr;
            a.x3 = (n_tile >= 4) ? (xt + 3*K) : nullptr;
            a.y0 = (n_tile >= 1) ? (yt + 0*(size_t)M + base_m) : nullptr;
            a.y1 = (n_tile >= 2) ? (yt + 1*(size_t)M + base_m) : nullptr;
            a.y2 = (n_tile >= 3) ? (yt + 2*(size_t)M + base_m) : nullptr;
            a.y3 = (n_tile >= 4) ? (yt + 3*(size_t)M + base_m) : nullptr;
            a.sum_x0 = sx[0]; a.sum_x1 = sx[1]; a.sum_x2 = sx[2]; a.sum_x3 = sx[3];
            a.gate0 = 1.f; a.gate1 = 1.f; a.gate2 = 1.f; a.gate3 = 1.f;
            a.gate0 = 1.f; a.gate1 = 1.f; a.gate2 = 1.f; a.gate3 = 1.f;
            a.wq = wq_packed + (size_t)bi * (size_t)ld_w_bytes;
            if (gran == quant_granularity_t::per_group) {
                alignas(64) float sc_buf[M_blk];
                alignas(64) int32_t zp_buf[M_blk];
                alignas(64) float b_buf[M_blk];
                const int gs = group_size > 0 ? group_size : M_blk;
                for (int m = 0; m < M_blk; ++m) {
                    const int g = (base_m + m) / gs;
                    sc_buf[m] = scales[g];
                    if (zps) zp_buf[m] = zps[g];
                    if (bias) b_buf[m] = bias[g];
                }
                a.scales = sc_buf;
                a.zps = zps ? zp_buf : nullptr;
                a.bias = bias ? b_buf : nullptr;
            } else {
                a.scales = get_sc_ptr(bi);
                a.zps = get_zp_ptr(bi);
                a.bias = get_bs_ptr(bi);
            }
            a.K = K; a.M_tail = 0; a.gran = static_cast<int>(gran);
            a.is_u8 = (wtype == w_dtype_t::u8) ? 1 : 0;
            a.w_nbits = w_nbits; a.w_unsigned = w_unsigned; a.k_step_bytes = k_step_bytes;
            fn(&a);
        }
        // Tail block
        if (M_tail) {
            const int bi = M_full;
            JitMiniGemmAvx512Fp32::CallArgs a{};
            const int base_m = bi * M_blk;
            a.n_cols = n_tile;
            a.x0 = (n_tile >= 1) ? (xt + 0*K) : nullptr;
            a.x1 = (n_tile >= 2) ? (xt + 1*K) : nullptr;
            a.x2 = (n_tile >= 3) ? (xt + 2*K) : nullptr;
            a.x3 = (n_tile >= 4) ? (xt + 3*K) : nullptr;
            a.y0 = (n_tile >= 1) ? (yt + 0*(size_t)M + base_m) : nullptr;
            a.y1 = (n_tile >= 2) ? (yt + 1*(size_t)M + base_m) : nullptr;
            a.y2 = (n_tile >= 3) ? (yt + 2*(size_t)M + base_m) : nullptr;
            a.y3 = (n_tile >= 4) ? (yt + 3*(size_t)M + base_m) : nullptr;
            a.sum_x0 = sx[0]; a.sum_x1 = sx[1]; a.sum_x2 = sx[2]; a.sum_x3 = sx[3];
            a.gate0 = 1.f; a.gate1 = 1.f; a.gate2 = 1.f; a.gate3 = 1.f;
            a.gate0 = 1.f; a.gate1 = 1.f; a.gate2 = 1.f; a.gate3 = 1.f;
            a.wq = wq_packed + (size_t)bi * (size_t)ld_w_bytes;
            if (gran == quant_granularity_t::per_group) {
                alignas(64) float sc_buf[M_blk];
                alignas(64) int32_t zp_buf[M_blk];
                alignas(64) float b_buf[M_blk];
                const int gs = group_size > 0 ? group_size : M_blk;
                for (int m = 0; m < M_blk; ++m) {
                    const int g = (base_m + m) / gs;
                    sc_buf[m] = scales[g];
                    if (zps) zp_buf[m] = zps[g];
                    if (bias) b_buf[m] = bias[g];
                }
                a.scales = sc_buf;
                a.zps = zps ? zp_buf : nullptr;
                a.bias = bias ? b_buf : nullptr;
            } else {
                a.scales = get_sc_ptr(bi);
                a.zps = get_zp_ptr(bi);
                a.bias = get_bs_ptr(bi);
            }
            a.K = K; a.M_tail = M_tail; a.gran = static_cast<int>(gran);
            a.is_u8 = (wtype == w_dtype_t::u8) ? 1 : 0;
            a.w_nbits = w_nbits; a.w_unsigned = w_unsigned; a.k_step_bytes = k_step_bytes;
            fn(&a);
        }
        n_done += n_tile;
    }
    return true;
}

} // namespace ov::intel_cpu::x64::gemmv_jit
