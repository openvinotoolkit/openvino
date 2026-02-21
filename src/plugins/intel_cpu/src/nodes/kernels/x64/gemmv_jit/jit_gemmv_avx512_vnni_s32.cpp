// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "jit_gemmv_avx512_vnni_s32.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "openvino/core/except.hpp"
#include "jit_prebuilt_pool.hpp"
#include "xbyak/xbyak_util.h"

namespace ov::intel_cpu::x64::gemmv_jit {
namespace {

constexpr int kBlock = 16;
constexpr int kGroupBytes = 64; // 16 lanes * 4 bytes per K-group
constexpr int kMainUnrollGroups = 16; // groups processed per main-loop iteration
constexpr int kPrefetchGroupSpan = 16; // how many K-groups ahead we prefetch
constexpr int kPrefetchWBytes = kGroupBytes * kPrefetchGroupSpan;
constexpr int kXAdvanceBytes = kMainUnrollGroups * 4; // bytes consumed from X per main trip
constexpr int kPrefetchXBytes = kXAdvanceBytes; // bytes ahead for X-prefetch (one iteration)

using CallArgs = JitGemmvAvx512VnniS32::CallArgs;

} // namespace

JitGemmvAvx512VnniS32::JitGemmvAvx512VnniS32()
    : dnnl::impl::cpu::x64::jit_generator_t("jit_gemmv_avx512_vnni_s32",
            dnnl::impl::cpu::x64::cpu_isa_t::avx512_core_vnni) {
    auto st = create_kernel();
    if (st != dnnl::impl::status::success) {
        OPENVINO_THROW("Failed to build jit_gemmv_avx512_vnni_s32 kernel");
    }
    fn_ = reinterpret_cast<fn_t>(jit_ker());
}

void JitGemmvAvx512VnniS32::generate() {
    using namespace Xbyak;
    setDefaultJmpNEAR(true);
#if defined(OPENVINO_ARCH_X86_64)
    endbr64();
#endif

    const Reg64 reg_args = rdi;
    const Reg64 reg_x = r8;
    const Reg64 reg_w = r9;
    const Reg64 reg_y = r10;
    const Reg64 reg_k_total = r11;
    const Reg64 reg_k_iter = r12;
    const Reg64 reg_k_main = r13;
    const Reg64 reg_k_bytes = r14;
    const Reg64 reg_sumw_ptr = r15;
    const Reg64 reg_K_actual = rsi;
    const Reg64 reg_need_sumw = rbx;
    const Reg64 reg_bias_lanes = rbp;

    const Zmm zAcc = zmm0;
    const Zmm zAccHi0 = zmm30;
    const Zmm zAccHi1 = zmm31;
    const Zmm zAccHi2 = zmm13;
    const Zmm zSumW = zmm1;
    const Zmm zOnes = zmm2;
    const Zmm zX0 = zmm3;
    const Zmm zX1 = zmm4;
    const Zmm zX2 = zmm5;
    const Zmm zX3 = zmm6;
    const Zmm zW0 = zmm7;
    const Zmm zW1 = zmm8;
    const Zmm zW2 = zmm9;
    const Zmm zW3 = zmm10;
    const Zmm zX4 = zmm22;
    const Zmm zX5 = zmm23;
    const Zmm zX6 = zmm24;
    const Zmm zX7 = zmm25;
    const Zmm zW4 = zmm26;
    const Zmm zW5 = zmm27;
    const Zmm zW6 = zmm28;
    const Zmm zW7 = zmm29;

    const Zmm zTmp0 = zmm11;
    const Zmm zTmp1 = zmm12;
    const Zmm zScale = zmm14;
    const Zmm zBias = zmm15;
    const Zmm zAccF = zmm16;
    const Zmm zCompF = zmm17;
    const Zmm zRes = zmm18;
    const Zmm zCompInt = zmm19;
    const Zmm zZpW = zmm20;
    const Zmm zSumWInt = zmm21;

    const Xmm xmmTmp = xmm11;

    push(rbx);
    push(rbp);
    push(r12);
    push(r13);
    push(r14);
    push(r15);

    // Early exit for dump-only path (used by bench diagnostics)
    {
        Label L_not_dump_only;
        mov(eax, dword[reg_args + offsetof(CallArgs, dbg_dump_only)]);
        test(eax, eax);
        jz(L_not_dump_only);
        pop(r15); pop(r14); pop(r13); pop(r12); pop(rbp); pop(rbx);
        ret();
        L(L_not_dump_only);
    }

    // Load pointers and scalars
    mov(reg_x, ptr[reg_args + offsetof(CallArgs, xq)]);
    mov(reg_w, ptr[reg_args + offsetof(CallArgs, wq)]);
    mov(reg_y, ptr[reg_args + offsetof(CallArgs, y)]);
    mov(eax, dword[reg_args + offsetof(CallArgs, K_groups)]);
    mov(reg_k_total, rax);
    mov(eax, dword[reg_args + offsetof(CallArgs, K_actual)]);
    mov(reg_K_actual, rax);
    mov(reg_sumw_ptr, ptr[reg_args + offsetof(CallArgs, sumW_lanes)]);
    mov(reg_bias_lanes, ptr[reg_args + offsetof(CallArgs, bias_lanes)]);

    xor_(reg_k_iter, reg_k_iter);
    xor_(reg_k_bytes, reg_k_bytes);
    mov(reg_k_main, reg_k_total);
    and_(reg_k_main, -kMainUnrollGroups);

    // Track whether we need to accumulate sumW
    xor_(reg_need_sumw, reg_need_sumw);
    mov(rax, reg_sumw_ptr);
    test(rax, rax);
    setz(al);
    movzx(reg_need_sumw, al);

    // Prepare masks: k1|k2 for M tail, k3 enabling sumW accumulation when needed
    const Reg32 reg_tail = edx;
    const Reg32 reg_mask_val = ecx;
    Label L_mask_ready;
    Label L_mask_done;
    mov(reg_tail, dword[reg_args + offsetof(CallArgs, M_tail)]);
    test(reg_tail, reg_tail);
    jz(L_mask_ready);
    mov(eax, 1);
    mov(reg_mask_val, reg_tail);
    shl(eax, cl);
    sub(eax, 1);
    kmovw(k1, eax);
    kmovw(k2, eax);
    jmp(L_mask_done);
    L(L_mask_ready);
    mov(eax, 0xFFFF);
    kmovw(k1, eax);
    kmovw(k2, eax);
    L(L_mask_done);

    kxorw(k3, k3, k3);
    Label L_need_sumw_mask_done;
    test(reg_need_sumw, reg_need_sumw);
    jz(L_need_sumw_mask_done);
    mov(eax, 0xFFFF);
    kmovw(k3, eax);
    L(L_need_sumw_mask_done);

    // Zero accumulators and prepare constants
    vpxord(zAcc, zAcc, zAcc);
    vpxord(zAccHi0, zAccHi0, zAccHi0);
    vpxord(zAccHi1, zAccHi1, zAccHi1);
    vpxord(zAccHi2, zAccHi2, zAccHi2);
    vpxord(zSumW, zSumW, zSumW);
    mov(eax, 0x01010101);
    movd(xmmTmp, eax);
    vpbroadcastd(zOnes, xmmTmp);

    // Main K-loop (unrolled by 16 groups)
    Label L_k_main, L_k_tail, L_k_done;
    L(L_k_main);
    cmp(reg_k_iter, reg_k_main);
    jge(L_k_tail);

    // Keep a steady stream of data flowing from memory.
    prefetcht0(ptr[reg_w + kPrefetchWBytes]);
    prefetcht0(ptr[reg_w + kPrefetchWBytes + kGroupBytes * 8]);
    prefetcht0(ptr[reg_x + kPrefetchXBytes]);

    // Preload weights for the first eight groups so vpdpbusd can fire back-to-back.
    vmovdqu8(zW0, ptr[reg_w]);
    vmovdqu8(zW1, ptr[reg_w + kGroupBytes]);
    vmovdqu8(zW2, ptr[reg_w + kGroupBytes * 2]);
    vmovdqu8(zW3, ptr[reg_w + kGroupBytes * 3]);
    vmovdqu8(zW4, ptr[reg_w + kGroupBytes * 4]);
    vmovdqu8(zW5, ptr[reg_w + kGroupBytes * 5]);
    vmovdqu8(zW6, ptr[reg_w + kGroupBytes * 6]);
    vmovdqu8(zW7, ptr[reg_w + kGroupBytes * 7]);

    // group 0
    vpbroadcastd(zX0, ptr[reg_x]);
    vpdpbusd(zAcc, zX0, zW0);
    vpdpbusd(zSumW | k3, zOnes, zW0);

    // group 1
    vpbroadcastd(zX1, ptr[reg_x + 4]);
    vpdpbusd(zAcc, zX1, zW1);
    vpdpbusd(zSumW | k3, zOnes, zW1);

    // group 2
    vpbroadcastd(zX2, ptr[reg_x + 8]);
    vpdpbusd(zAcc, zX2, zW2);
    vpdpbusd(zSumW | k3, zOnes, zW2);

    // group 3
    vpbroadcastd(zX3, ptr[reg_x + 12]);
    vpdpbusd(zAcc, zX3, zW3);
    vpdpbusd(zSumW | k3, zOnes, zW3);

    // group 4
    vpbroadcastd(zX4, ptr[reg_x + 16]);
    vpdpbusd(zAccHi0, zX4, zW4);
    vpdpbusd(zSumW | k3, zOnes, zW4);

    // group 5
    vpbroadcastd(zX5, ptr[reg_x + 20]);
    vpdpbusd(zAccHi0, zX5, zW5);
    vpdpbusd(zSumW | k3, zOnes, zW5);

    // group 6
    vpbroadcastd(zX6, ptr[reg_x + 24]);
    vpdpbusd(zAccHi0, zX6, zW6);
    vpdpbusd(zSumW | k3, zOnes, zW6);

    // group 7
    vpbroadcastd(zX7, ptr[reg_x + 28]);
    vpdpbusd(zAccHi0, zX7, zW7);
    vpdpbusd(zSumW | k3, zOnes, zW7);

    // Preload weights for the next eight groups
    vmovdqu8(zW0, ptr[reg_w + kGroupBytes * 8]);
    vmovdqu8(zW1, ptr[reg_w + kGroupBytes * 9]);
    vmovdqu8(zW2, ptr[reg_w + kGroupBytes * 10]);
    vmovdqu8(zW3, ptr[reg_w + kGroupBytes * 11]);
    vmovdqu8(zW4, ptr[reg_w + kGroupBytes * 12]);
    vmovdqu8(zW5, ptr[reg_w + kGroupBytes * 13]);
    vmovdqu8(zW6, ptr[reg_w + kGroupBytes * 14]);
    vmovdqu8(zW7, ptr[reg_w + kGroupBytes * 15]);

    // group 8
    vpbroadcastd(zX0, ptr[reg_x + 32]);
    vpdpbusd(zAccHi1, zX0, zW0);
    vpdpbusd(zSumW | k3, zOnes, zW0);

    // group 9
    vpbroadcastd(zX1, ptr[reg_x + 36]);
    vpdpbusd(zAccHi1, zX1, zW1);
    vpdpbusd(zSumW | k3, zOnes, zW1);

    // group 10
    vpbroadcastd(zX2, ptr[reg_x + 40]);
    vpdpbusd(zAccHi1, zX2, zW2);
    vpdpbusd(zSumW | k3, zOnes, zW2);

    // group 11
    vpbroadcastd(zX3, ptr[reg_x + 44]);
    vpdpbusd(zAccHi1, zX3, zW3);
    vpdpbusd(zSumW | k3, zOnes, zW3);

    // group 12
    vpbroadcastd(zX4, ptr[reg_x + 48]);
    vpdpbusd(zAccHi2, zX4, zW4);
    vpdpbusd(zSumW | k3, zOnes, zW4);

    // group 13
    vpbroadcastd(zX5, ptr[reg_x + 52]);
    vpdpbusd(zAccHi2, zX5, zW5);
    vpdpbusd(zSumW | k3, zOnes, zW5);

    // group 14
    vpbroadcastd(zX6, ptr[reg_x + 56]);
    vpdpbusd(zAccHi2, zX6, zW6);
    vpdpbusd(zSumW | k3, zOnes, zW6);

    // group 15
    vpbroadcastd(zX7, ptr[reg_x + 60]);
    vpdpbusd(zAccHi2, zX7, zW7);
    vpdpbusd(zSumW | k3, zOnes, zW7);

    add(reg_x, kXAdvanceBytes);
    add(reg_w, kGroupBytes * kMainUnrollGroups);
    add(reg_k_iter, kMainUnrollGroups);
    add(reg_k_bytes, kXAdvanceBytes);
    jmp(L_k_main);

    // Tail loop (handle remaining <4 groups, including partial K)
    L(L_k_tail);
    cmp(reg_k_iter, reg_k_total);
    jge(L_k_done);

    Label L_tail_loop;
    L(L_tail_loop);
    cmp(reg_k_iter, reg_k_total);
    jge(L_k_done);

    Label L_tail_partial;
    Label L_tail_loaded;
    vmovdqu8(zW0, ptr[reg_w]);
    mov(rcx, reg_K_actual);
    sub(rcx, reg_k_bytes);
    cmp(rcx, 4);
    jl(L_tail_partial);
    // full 4-byte load
    vmovd(xmmTmp, ptr[reg_x]);
    jmp(L_tail_loaded);

    // partial load path
    L(L_tail_partial);
    xor_(eax, eax);
    mov(rdx, reg_x);
    movzx(eax, byte[rdx]);
    cmp(rcx, 1);
    je(L_tail_loaded);
    movzx(edx, byte[rdx + 1]);
    shl(edx, 8);
    or_(eax, edx);
    cmp(rcx, 2);
    je(L_tail_loaded);
    movzx(edx, byte[rdx + 2]);
    shl(edx, 16);
    or_(eax, edx);
    movd(xmmTmp, eax);

    L(L_tail_loaded);
    vpbroadcastd(zX0, xmmTmp);
    vpdpbusd(zAcc, zX0, zW0);
    vpdpbusd(zSumW | k3, zOnes, zW0);

    add(reg_x, 4);
    add(reg_w, kGroupBytes);
    inc(reg_k_iter);
    add(reg_k_bytes, 4);
    jmp(L_tail_loop);

    L(L_k_done);

    // Merge high accumulator lanes before debug/epilogue processing
    vpaddd(zAcc, zAcc, zAccHi0);
    vpaddd(zAccHi1, zAccHi1, zAccHi2);
    vpaddd(zAcc, zAcc, zAccHi1);

    // Debug capture (raw s32 accumulators and sumW)
    {
        Label L_no_dbg;
        mov(eax, dword[reg_args + offsetof(CallArgs, dbg_enable)]);
        test(eax, eax);
        jz(L_no_dbg);

        Label L_skip_dbg_acc;
        Label L_skip_dbg_sumw;
        mov(rax, ptr[reg_args + offsetof(CallArgs, dbg_acc)]);
        test(rax, rax);
        jz(L_skip_dbg_acc);
        vmovdqu32(ptr[rax] | k2, zAcc);
        L(L_skip_dbg_acc);

        mov(rax, ptr[reg_args + offsetof(CallArgs, dbg_sumw)]);
        test(rax, rax);
        jz(L_skip_dbg_sumw);
        vmovdqu32(ptr[rax] | k2, zSumW);
        L(L_skip_dbg_sumw);

        mov(eax, dword[reg_args + offsetof(CallArgs, dbg_dump_only)]);
        test(eax, eax);
        jz(L_no_dbg);
        pop(r15); pop(r14); pop(r13); pop(r12); pop(rbp); pop(rbx);
        ret();
        L(L_no_dbg);
    }

    // Convert accumulators and prepare per-lane metadata
    vcvtdq2ps(zAccF, zAcc);

    // sumW lanes: prefer precomputed pointer, otherwise use accumulated values
    {
        Label L_use_runtime_sumw;
        Label L_sumw_ready;
        mov(rax, reg_sumw_ptr);
        test(rax, rax);
        jz(L_use_runtime_sumw);
        vmovdqu32(zSumWInt | k1 | T_z, ptr[rax]);
        jmp(L_sumw_ready);
        L(L_use_runtime_sumw);
        vmovdqa32(zSumWInt, zSumW);
        L(L_sumw_ready);
    }

    // (-zp_x) * sumW term
    mov(eax, dword[reg_args + offsetof(CallArgs, zp_x)]);
    neg(eax);
    movd(xmmTmp, eax);
    vpbroadcastd(zTmp0, xmmTmp);
    vpmulld(zCompInt, zSumWInt, zTmp0);

    // base term (K*zp_x - sum_x_q)
    mov(eax, dword[reg_args + offsetof(CallArgs, zp_x)]);
    mov(edx, dword[reg_args + offsetof(CallArgs, K_actual)]);
    imul(edx, eax);
    mov(ecx, dword[reg_args + offsetof(CallArgs, sum_x_q)]);
    sub(edx, ecx);
    movd(xmmTmp, edx);
    vpbroadcastd(zTmp0, xmmTmp);

    // zp_w (lane or scalar)
    {
        Label L_use_scalar_zp;
        Label L_zp_ready;
        mov(rax, ptr[reg_args + offsetof(CallArgs, zpw_lanes)]);
        test(rax, rax);
        jz(L_use_scalar_zp);
        vmovdqu32(zZpW | k1 | T_z, ptr[rax]);
        jmp(L_zp_ready);
        L(L_use_scalar_zp);
        mov(eax, dword[reg_args + offsetof(CallArgs, zp_w)]);
        movd(xmmTmp, eax);
        vpbroadcastd(zZpW, xmmTmp);
        L(L_zp_ready);
    }

    vpmulld(zTmp1, zZpW, zTmp0);
    vpaddd(zCompInt, zCompInt, zTmp1);
    vcvtdq2ps(zCompF, zCompInt);

    // s_w (lane or scalar) * s_x
    {
        Label L_use_scalar_scale;
        Label L_scale_ready;
        mov(rax, ptr[reg_args + offsetof(CallArgs, sw_lanes)]);
        test(rax, rax);
        jz(L_use_scalar_scale);
        vmovups(zScale | k1 | T_z, ptr[rax]);
        jmp(L_scale_ready);
        L(L_use_scalar_scale);
        vbroadcastss(zScale, ptr[reg_args + offsetof(CallArgs, s_w)]);
        L(L_scale_ready);
    }
    vbroadcastss(zTmp0, ptr[reg_args + offsetof(CallArgs, s_x)]);
    vmulps(zScale, zScale, zTmp0);

    vmulps(zAccF, zAccF, zScale);
    vmulps(zCompF, zCompF, zScale);
    vaddps(zRes, zAccF, zCompF);

    // Bias (lane or scalar)
    {
        Label L_use_scalar_bias;
        Label L_bias_ready;
        mov(rax, reg_bias_lanes);
        test(rax, rax);
        jz(L_use_scalar_bias);
        vmovups(zBias | k1 | T_z, ptr[rax]);
        jmp(L_bias_ready);
        L(L_use_scalar_bias);
        vbroadcastss(zBias, ptr[reg_args + offsetof(CallArgs, bias)]);
        L(L_bias_ready);
    }
    vaddps(zRes, zRes, zBias);

    // Accumulate into Y if requested
    {
        Label L_no_accum;
        mov(eax, dword[reg_args + offsetof(CallArgs, accumulate)]);
        test(eax, eax);
        jz(L_no_accum);
        vmovups(zTmp0 | k1 | T_z, ptr[reg_y]);
        vaddps(zRes, zRes, zTmp0);
        L(L_no_accum);
    }

    // Store results (masked tail if needed)
    {
        Label L_tail_store, L_store_done;
        mov(eax, dword[reg_args + offsetof(CallArgs, M_tail)]);
        test(eax, eax);
        jz(L_tail_store);
        vmovups(ptr[reg_y] | k1, zRes);
        jmp(L_store_done);
        L(L_tail_store);
        vmovups(ptr[reg_y], zRes);
        L(L_store_done);
    }

    pop(r15); pop(r14); pop(r13); pop(r12); pop(rbp); pop(rbx);
    ret();
}

void JitGemmvAvx512VnniKernel::operator()(const gemmv_ukr_params_t* p) const {
    OPENVINO_ASSERT(p, "VNNI kernel received null params");
    OPENVINO_ASSERT(p->x_q8 && p->wq && p->y, "VNNI kernel requires quantized X, W, and Y buffers");

    auto fn = jit_prebuilt_pool::get_typed<JitGemmvAvx512VnniS32::fn_t>(kernel_kind::vnni_int8);
    OPENVINO_ASSERT(fn != nullptr, "VNNI kernel JIT was not initialized");

    const int blocks = (p->M + kBlock - 1) / kBlock;
    const int K_groups = (p->K + 3) / 4;
    float* y_base = static_cast<float*>(p->y);
    const int base_m = p->m_base;

    for (int bi = 0; bi < blocks; ++bi) {
        const int m0 = bi * kBlock;
        const int valid = std::min(kBlock, p->M - m0);
        if (valid <= 0) {
            break;
        }

        CallArgs args{};
        args.xq = p->x_q8;
        args.K_groups = K_groups;
        args.K_actual = p->K;
        args.wq = p->wq + static_cast<size_t>(bi) * static_cast<size_t>(p->ld_w_bytes);
        args.ld_w_gbytes = p->ld_w_bytes;
        args.y = y_base + m0;
        args.M_tail = (valid == kBlock) ? 0 : valid;
        args.s_w = p->scales ? p->scales[0] : 1.f;
        args.s_x = p->x_scale;
        args.zp_w = p->zps ? p->zps[0] : 0;
        args.zp_x = p->x_zp;
        args.sum_x_q = p->sum_x_q;
        args.bias = (p->bias && p->gran == quant_granularity_t::per_tensor) ? p->bias[0] : 0.f;
        args.accumulate = p->accumulate ? 1 : 0;
        args.sw_lanes = p->lane_scales ? (p->lane_scales + base_m + m0) : nullptr;
        args.bias_lanes = p->lane_bias ? (p->lane_bias + base_m + m0) : nullptr;
        args.zpw_lanes = p->lane_zps ? (p->lane_zps + base_m + m0) : nullptr;
        args.sumW_lanes = p->sumW_precomp ? (p->sumW_precomp + base_m + m0) : nullptr;
        args.dbg_enable = 0;
        args.dbg_acc = nullptr;
        args.dbg_sumw = nullptr;
        args.dbg_dump_only = 0;

        fn(&args);
    }
}

bool run_gemmv_vnni_i8u8_fp32(const uint8_t* xq, int K,
                              const uint8_t* wq_k4, int M, int ld_w_gbytes,
                              float s_w, int32_t zp_w, float s_x, int32_t zp_x,
                              float* y, float bias,
                              const int32_t* sumW_precomp,
                              int dbg_block, int32_t* dbg_acc, int32_t* dbg_sumw,
                              int dbg_dump_only) {
    Xbyak::util::Cpu cpu;
    if (!cpu.has(Xbyak::util::Cpu::tAVX512_VNNI) || !cpu.has(Xbyak::util::Cpu::tAVX512BW)) {
        return false;
    }

    auto fn = jit_prebuilt_pool::get_typed<JitGemmvAvx512VnniS32::fn_t>(kernel_kind::vnni_int8);
    if (!fn) {
        return false;
    }

    const int K_groups = (K + 3) / 4;
    const int blocks = (M + kBlock - 1) / kBlock;
    int32_t sum_x_q = 0;
    for (int i = 0; i < K; ++i) {
        sum_x_q += static_cast<int32_t>(xq[i]);
    }

    for (int bi = 0; bi < blocks; ++bi) {
        const int m0 = bi * kBlock;
        const int valid = std::min(kBlock, M - m0);
        if (valid <= 0) {
            break;
        }

        const bool dbg_on = (dbg_block >= 0) && (dbg_block == bi) && dbg_acc && dbg_sumw;
        const bool dump_only = dbg_on && (dbg_dump_only != 0);

        CallArgs args{};
        args.xq = xq;
        args.K_groups = dump_only ? 0 : K_groups;
        args.K_actual = K;
        args.wq = wq_k4 + static_cast<size_t>(bi) * static_cast<size_t>(ld_w_gbytes);
        args.ld_w_gbytes = ld_w_gbytes;
        args.y = y + m0;
        args.M_tail = (valid == kBlock) ? 0 : valid;
        args.s_w = s_w;
        args.s_x = s_x;
        args.zp_w = zp_w;
        args.zp_x = zp_x;
        args.sum_x_q = sum_x_q;
        args.bias = bias;
        args.accumulate = 0;
        args.sw_lanes = nullptr;
        args.bias_lanes = nullptr;
        args.zpw_lanes = nullptr;
        args.sumW_lanes = sumW_precomp ? (sumW_precomp + m0) : nullptr;

        args.dbg_enable = dbg_on ? 1 : 0;
        args.dbg_acc = dbg_on ? dbg_acc : nullptr;
        args.dbg_sumw = dbg_on ? dbg_sumw : nullptr;
        args.dbg_dump_only = dump_only ? 1 : 0;

        fn(&args);
    }
    return true;
}

} // namespace ov::intel_cpu::x64::gemmv_jit
