// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "jit_gemmv_avx512_vnni_s32.hpp"
#include "xbyak/xbyak_util.h"

namespace ov::intel_cpu::x64::gemmv_jit {

JitGemmvAvx512VnniS32::JitGemmvAvx512VnniS32() : Xbyak::CodeGenerator(8 * 1024) {
    using namespace Xbyak;
    setDefaultJmpNEAR(true);

    const Reg64 reg_args = rdi;
    const Reg64 reg_x = r8;
    const Reg64 reg_w = r9;
    const Reg64 reg_y = r10;
    const Reg64 reg_k = r11;
    const Reg64 reg_k_iter = r12;
    const Reg64 reg_xg = r15; // advancing X-group pointer

    // gp temps; keep 16-byte stack alignment
    push(rbx); push(r12); push(r13); push(r14); push(r15);

    // load args
    mov(reg_x, ptr[reg_args + offsetof(CallArgs, xq)]);
    mov(reg_w, ptr[reg_args + offsetof(CallArgs, wq)]);
    mov(reg_y, ptr[reg_args + offsetof(CallArgs, y)]);
    mov(eax, dword[reg_args + offsetof(CallArgs, K_groups)]); mov(reg_k, rax);
    mov(reg_xg, reg_x);

    // zmm regs
    const Zmm zAcc = zmm0;
    const Zmm zSumW = zmm1;
    const Zmm zW = zmm2;           // s8 bytes
    const Zmm zX = zmm3;           // u8 bytes by 4 broadcasted per dword
    const Zmm zTmp = zmm4;
    const Zmm zTmp2 = zmm5;
    const Zmm zF0 = zmm6;          // fp32 tmp
    const Zmm zF1 = zmm7;          // fp32 tmp
    const Zmm zFbias = zmm8;
    const Zmm zFswsx = zmm9;       // s_w * s_x broadcast
    const Zmm zFcomp = zmm10;      // compensation vector
    const Zmm zFsumW = zmm11;      // sumW (fp32)

    // Early minimal dump-only path: if dbg_dump_only==1, return immediately (no vector ops)
    {
        Label L_not_dump_only;
        mov(eax, dword[reg_args + offsetof(CallArgs, dbg_dump_only)]);
        test(eax, eax); jz(L_not_dump_only);
        pop(r15); pop(r14); pop(r13); pop(r12); pop(rbx);
        ret();
        L(L_not_dump_only);
    }

    // init accumulators
    vpxord(zAcc, zAcc, zAcc);
    vpxord(zSumW, zSumW, zSumW);

    // prepare constants
    // zFswsx = s_w * s_x
    vbroadcastss(zFswsx, ptr[reg_args + offsetof(CallArgs, s_w)]);
    vbroadcastss(zTmp, ptr[reg_args + offsetof(CallArgs, s_x)]);
    vmulps(zFswsx, zFswsx, zTmp);
    // zFbias = bias
    vbroadcastss(zFbias, ptr[reg_args + offsetof(CallArgs, bias)]);

    // (sumW path temporarily disabled in minimal loop)

    // scalar common compensation term: (-zp_w)*sum_x_q + K*zp_w*zp_x
    // load zp_w -> eax, zp_x -> edx, sum_x_q -> ecx, K_groups -> esi
    mov(eax, dword[reg_args + offsetof(CallArgs, zp_w)]);
    mov(edx, dword[reg_args + offsetof(CallArgs, zp_x)]);
    mov(ecx, dword[reg_args + offsetof(CallArgs, sum_x_q)]);
    mov(esi, dword[reg_args + offsetof(CallArgs, K_groups)]);
    // K = K_groups * 4 in ebx
    mov(ebx, esi); lea(ebx, ptr[ebx * 4]);
    // t1 = K * zp_w * zp_x in edi
    mov(edi, eax); imul(edi, edx); imul(edi, ebx);
    // t2 = - zp_w * sum_x_q in ebx
    mov(ebx, eax); imul(ebx, ecx); neg(ebx);
    // sc = t1 + t2 in ebx
    add(ebx, edi);
    // broadcast sc as fp32 in zFcomp (we'll add sumW*-x_zp later)
    movd(xmm1, ebx); pshufd(xmm1, xmm1, 0); vcvtdq2ps(zFcomp, xmm1);

    // K loop
    xor_(reg_k_iter, reg_k_iter);
    Label L_k, L_k_done;
    L(L_k);
    cmp(reg_k_iter, reg_k);
    jge(L_k_done);
    // load W group (64B): 16 rows x 4 bytes (single 64B zmm load needs AVX512BW)
    vmovdqu8(zW, ptr[reg_w]);
    add(reg_w, 64);
    // load X group 4 bytes -> broadcast per dword
    mov(ecx, dword[reg_xg]);
    add(reg_xg, 4);
    movd(xmm0, ecx); vpbroadcastd(zX, xmm0);
    // acc += X(u8) * W(s8)
    vpdpbusd(zAcc, zX, zW);
    // (sumW accumulation disabled in minimal loop)

    inc(reg_k_iter);
    jmp(L_k);
    L(L_k_done);

    // convert accum to fp32: zF0 = float(zAcc)
    // Optional debug capture of s32 accumulators (acc and sumW)
    {
        Label L_no_dbg;
        mov(eax, dword[reg_args + offsetof(CallArgs, dbg_enable)]);
        test(eax, eax); jz(L_no_dbg);
        // store acc
        mov(rax, ptr[reg_args + offsetof(CallArgs, dbg_acc)]);
        test(rax, rax); jz(L_no_dbg);
        {
            Label L_full, L_done;
            mov(eax, dword[reg_args + offsetof(CallArgs, M_tail)]);
            test(eax, eax); jz(L_full);
            mov(ecx, eax); mov(eax, 1); shl(eax, cl); sub(eax, 1); kmovw(k1, eax);
            vmovdqu32(ptr[rax] | k1, zAcc);
            jmp(L_done);
            L(L_full);
            vmovdqu32(ptr[rax], zAcc);
            L(L_done);
        }
        // store sumW
        mov(rax, ptr[reg_args + offsetof(CallArgs, dbg_sumw)]);
        test(rax, rax); jz(L_no_dbg);
        {
            Label L_full2, L_done2;
            mov(eax, dword[reg_args + offsetof(CallArgs, M_tail)]);
            test(eax, eax); jz(L_full2);
            mov(ecx, eax); mov(eax, 1); shl(eax, cl); sub(eax, 1); kmovw(k1, eax);
            vmovdqu32(ptr[rax] | k1, zSumW);
            jmp(L_done2);
            L(L_full2);
            vmovdqu32(ptr[rax], zSumW);
            L(L_done2);
        }
        // if dump_only, return early
        mov(eax, dword[reg_args + offsetof(CallArgs, dbg_dump_only)]);
        test(eax, eax); jz(L_no_dbg);
        pop(r15); pop(r14); pop(r13); pop(r12); pop(rbx);
        ret();
        L(L_no_dbg);
    }

    // convert accum to fp32: zF0 = float(zAcc)
    vcvtdq2ps(zF0, zAcc);
    // per-tensor compensation part using sumW and x_zp
    {
        // zFsumW = float(sumW)
        vcvtdq2ps(zFsumW, zSumW);
        // zTmp = float(-x_zp)
        mov(eax, dword[reg_args + offsetof(CallArgs, zp_x)]);
        neg(eax);
        movd(xmm0, eax); pshufd(xmm0, xmm0, 0); vcvtdq2ps(zTmp, xmm0);
        // zFcomp += zFsumW * (-x_zp)
        vfmadd231ps(zFcomp, zFsumW, zTmp);
        // scale by s_w*s_x
        vmulps(zFcomp, zFcomp, zFswsx);
    }
    // y = zF0 * (s_w*s_x) + zFcomp + bias
    vmulps(zF0, zF0, zFswsx);
    vaddps(zF0, zF0, zFcomp);
    vaddps(zF0, zF0, zFbias);

    // store with tail mask if needed
    {
        Label L_full, L_done;
        mov(eax, dword[reg_args + offsetof(CallArgs, M_tail)]);
        test(eax, eax); jz(L_full);
        mov(ecx, eax); mov(eax, 1); shl(eax, cl); sub(eax, 1); kmovw(k1, eax);
        vmovups(ptr[reg_y] | k1, zF0);
        jmp(L_done);
        L(L_full);
        vmovups(ptr[reg_y], zF0);
        L(L_done);
    }

    pop(r15); pop(r14); pop(r13); pop(r12); pop(rbx);
    ret();

    ready();
}

bool run_gemmv_vnni_i8u8_fp32(const uint8_t* xq, int K,
                              const uint8_t* wq_k4, int M, int ld_w_gbytes,
                              float s_w, int32_t zp_w, float s_x, int32_t zp_x,
                              float* y, float bias,
                              int dbg_block, int32_t* dbg_acc, int32_t* dbg_sumw,
                              int dbg_dump_only) {
    Xbyak::util::Cpu cpu;
    if (!cpu.has(Xbyak::util::Cpu::tAVX512_VNNI)) return false;
    const int M_blk = 16;
    const int K_grp = (K + 3)/4;
    static JitGemmvAvx512VnniS32 jit;
    auto fn = jit.get();
    if (!fn) {
        return false;
    }
    const int M_full = M / M_blk;
    const int M_tail = M % M_blk;
    // precompute sum_x_q
    int32_t sum_x_q = 0; for (int k = 0; k < K; ++k) sum_x_q += (int)xq[k];
    // per-tensor only for now
    for (int bi = 0; bi < M_full; ++bi) {
        if (dbg_dump_only && bi != dbg_block) continue;
        JitGemmvAvx512VnniS32::CallArgs a{};
        a.xq = xq; a.wq = wq_k4 + (size_t)bi * (size_t)ld_w_gbytes; a.K_groups = K_grp; a.ld_w_gbytes = ld_w_gbytes;
        a.y = y + bi * M_blk; a.M_tail = 0; a.s_w = s_w; a.s_x = s_x; a.zp_w = zp_w; a.zp_x = zp_x; a.sum_x_q = sum_x_q; a.bias = bias;
        a.dbg_enable = (dbg_acc && dbg_sumw && bi == dbg_block) ? 1 : 0;
        a.dbg_acc = a.dbg_enable ? dbg_acc : nullptr;
        a.dbg_sumw = a.dbg_enable ? dbg_sumw : nullptr;
        a.dbg_dump_only = a.dbg_enable ? dbg_dump_only : 0;
        if (dbg_dump_only) a.K_groups = 0; // skip loop entirely in dump-only minimal repro
        fn(&a);
    }
    if (M_tail && (!dbg_dump_only || dbg_block == M_full)) {
        JitGemmvAvx512VnniS32::CallArgs a{};
        const int bi = M_full; a.xq = xq; a.wq = wq_k4 + (size_t)bi * (size_t)ld_w_gbytes; a.K_groups = K_grp; a.ld_w_gbytes = ld_w_gbytes;
        a.y = y + bi * M_blk; a.M_tail = M_tail; a.s_w = s_w; a.s_x = s_x; a.zp_w = zp_w; a.zp_x = zp_x; a.sum_x_q = sum_x_q; a.bias = bias;
        a.dbg_enable = (dbg_acc && dbg_sumw && bi == dbg_block) ? 1 : 0;
        a.dbg_acc = a.dbg_enable ? dbg_acc : nullptr;
        a.dbg_sumw = a.dbg_enable ? dbg_sumw : nullptr;
        a.dbg_dump_only = a.dbg_enable ? dbg_dump_only : 0;
        fn(&a);
    }
    return true;
}

} // namespace ov::intel_cpu::x64::gemmv_jit
