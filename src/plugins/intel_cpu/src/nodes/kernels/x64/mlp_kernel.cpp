// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlp_kernel.hpp"
#include "emitters/plugin/x64/jit_dnnl_emitters.hpp"
#include "mlp_utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu::x64;

using TileConfig = ov::Extensions::Cpu::TileConfig;
using TileConfiger = ov::Extensions::Cpu::TileConfiger;

namespace ov {
namespace intel_cpu {

void MKernel::generate() {
    Xbyak::Reg64 reg_A_addr = abi_param2;
    Xbyak::Reg64 reg_A_stride = abi_param3;
    Xbyak::Reg64 reg_B_addr = abi_param4;
    Xbyak::Reg64 reg_C_addr = abi_param5;
    Xbyak::Reg64 reg_C_stride = abi_param6;

    Xbyak::Reg64 reg_ktiles = rax;
    Xbyak::Reg64 reg_B_stride = r10;
    Xbyak::Reg64 reg_A1_addr = r11;
    Xbyak::Reg64 reg_prefetch = r12;

    Xbyak::Tmm tmmC00 = tmm0;
    Xbyak::Tmm tmmC01 = tmm1;
    Xbyak::Tmm tmmC10 = tmm2;
    Xbyak::Tmm tmmC11 = tmm3;
    Xbyak::Tmm tmmA0 = tmm4;
    Xbyak::Tmm tmmA1 = tmm5;
    Xbyak::Tmm tmmB0 = tmm6;
    Xbyak::Tmm tmmB1 = tmm7;

    auto num_PFB = m_prefetch_Blines;
    int cur_PFB = 0;

    Xbyak::Label loop_over_ktiles;
    Xbyak::Label skip_load;

    push(reg_prefetch);
    {
        auto reg_tmp = reg_B_stride;
        tilezero(tmmC00);
        tilezero(tmmC01);
        tilezero(tmmC10);
        tilezero(tmmC11);

        mov(reg_A_addr, ptr[abi_param1 + offsetof(call_args, pA)]);
        mov(reg_A_stride, ptr[abi_param1 + offsetof(call_args, strideA)]);
        mov(reg_B_addr, ptr[abi_param1 + offsetof(call_args, pB)]);
        mov(reg_C_addr, ptr[abi_param1 + offsetof(call_args, pC)]);
        mov(reg_C_stride, ptr[abi_param1 + offsetof(call_args, strideC)]);
        mov(reg_prefetch, ptr[abi_param1 + offsetof(call_args, prefetch)]);
        mov(reg_ktiles, ptr[abi_param1 + offsetof(call_args, k_tiles)]);

        lea(reg_A1_addr, ptr[reg_A_addr + reg_A_stride * 8]);
        lea(reg_A1_addr, ptr[reg_A1_addr + reg_A_stride * 8]);

        // reg_A1_addr = reg_A_addr if M <= 16 (to avoid tileloadd segmentfault)
        mov(reg_tmp, ptr[abi_param1 + offsetof(call_args, M)]);
        cmp(reg_tmp, 16);
        cmovle(reg_A1_addr, reg_A_addr);

        mov(reg_tmp, ptr[abi_param1 + offsetof(call_args, do_accumulation)]);
        and_(reg_tmp, 1);
        jz(skip_load);
        {
            auto reg_C1_addr = reg_tmp;
            tileloadd(tmmC00, ptr[reg_C_addr + reg_C_stride]);
            tileloadd(tmmC01, ptr[reg_C_addr + reg_C_stride + 64]);
            lea(reg_C1_addr, ptr[reg_C_addr + reg_C_stride * 8]);
            lea(reg_C1_addr, ptr[reg_C1_addr + reg_C_stride * 8]);
            tileloadd(tmmC10, ptr[reg_C1_addr + reg_C_stride]);
            tileloadd(tmmC11, ptr[reg_C1_addr + reg_C_stride + 64]);
        }
        L(skip_load);
    }

    mov(reg_B_stride, 64);

    auto const_A_steps = 64;

    align(64, false);
    L(loop_over_ktiles);
    {
        //                B: 1x2 tiles
        // A : 2x1 tiles  C: 2x2 tiles
        tileloadd(tmmA0, ptr[reg_A_addr + reg_A_stride]);
        tileloadd(tmmB0, ptr[reg_B_addr + reg_B_stride]);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);

        tdpbf16ps(tmmC00, tmmA0, tmmB0);
        if (cur_PFB < num_PFB) {
            prefetcht2(ptr[reg_prefetch + cur_PFB * 64]);
            cur_PFB++;
        }

        tileloadd(tmmA1, ptr[reg_A1_addr + reg_A_stride]);
        tdpbf16ps(tmmC10, tmmA1, tmmB0);
        if (cur_PFB < num_PFB) {
            prefetcht2(ptr[reg_prefetch + cur_PFB * 64]);
            cur_PFB++;
        }

        tileloadd(tmmB1, ptr[reg_B_addr + reg_B_stride]);
        tdpbf16ps(tmmC01, tmmA0, tmmB1);
        if (cur_PFB < num_PFB) {
            prefetcht2(ptr[reg_prefetch + cur_PFB * 64]);
            cur_PFB++;
        }

        tdpbf16ps(tmmC11, tmmA1, tmmB1);
        if (cur_PFB < num_PFB) {
            for (int pi = cur_PFB; pi < num_PFB; pi++) {
                prefetcht2(ptr[reg_prefetch + pi * 64]);
            }
        }

        lea(reg_prefetch, ptr[reg_prefetch + 64 * num_PFB]);
        lea(reg_A_addr, ptr[reg_A_addr + const_A_steps]);
        lea(reg_A1_addr, ptr[reg_A1_addr + const_A_steps]);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);
    }
    dec(reg_ktiles);
    jnz(loop_over_ktiles, T_NEAR);

    tilestored(ptr[reg_C_addr + reg_C_stride], tmmC00);
    tilestored(ptr[reg_C_addr + reg_C_stride + 64], tmmC01);
    lea(reg_C_addr, ptr[reg_C_addr + reg_C_stride * 8]);
    lea(reg_C_addr, ptr[reg_C_addr + reg_C_stride * 8]);
    tilestored(ptr[reg_C_addr + reg_C_stride], tmmC10);
    tilestored(ptr[reg_C_addr + reg_C_stride + 64], tmmC11);

    pop(reg_prefetch);
    ret();
}

void MKernel::tile_config_M(TileConfig& tile_cfg, int M) {
    auto rows0 = 16;
    auto rows1 = 16;
    if (M < 32) {
        // kernel is for processing Mtails
        if (M > 16) {
            rows0 = 16;
            rows1 = M - 16;
        } else {
            //  both A0 & A1 load from same memory, to avoid code-regeneration
            rows0 = rows1 = M;
        }
    }
    tile_cfg.reset(1,
                    0,
                    {
                        {rows0, 64},  // C00:0
                        {rows0, 64},  // C01:1
                        {rows1, 64},  // C10:2
                        {rows1, 64},  // C11:3
                        {rows0, 64},  // A0:4
                        {rows1, 64},  // A1:5
                        {16, 64},     // B0:6
                        {16, 64},     // B1:7
                    });
}

template <typename T>
void MKernel::repackB(ov::bfloat16* dst, T* src, int N_stride, int N, int K) {
    if (N == 16 && K == 32 && std::is_same<T, ov::bfloat16>::value) {
        // SIMD optimized version
        ov::Extensions::Cpu::XARCH::llm_mlp_transpose_epi32_16x16(dst, src, N_stride * sizeof(T));
        return;
    }

    assert(K <= 32);
    assert(N <= 16);
    int k = 0;
    ov::bfloat16 bf16zero(0.0f);
    for (; k < 32; k += 2) {
        int n = 0;
        bool is_k0_valid = (k) < K;
        bool is_k1_valid = (k + 1) < K;
        auto* psrc = src + k;
        for (; n < 16 && n < N; n++, psrc += N_stride) {
            *dst++ = is_k0_valid ? ov::bfloat16(psrc[0]) : bf16zero;
            *dst++ = is_k1_valid ? ov::bfloat16(psrc[1]) : bf16zero;
        }
        for (; n < 16; n++) {
            *dst++ = 0;
            *dst++ = 0;
        }
    }
}

template <typename T>
void MKernel::prepareB(PlainTensor& ret, T* p_weight, int stride, int N, int K) {
    OPENVINO_ASSERT((N % 32) == 0);
    OPENVINO_ASSERT((K % 32) == 0);
    // weight matrix is in unit of [N/32, Kx32]
    ret.resize<ov::bfloat16>({static_cast<size_t>(N / 32), static_cast<size_t>(K * 32)});

    auto N_stride = stride / sizeof(T);
    for (int n = 0, blkn = 0; n < N; n += 32, blkn++) {
        for (int k = 0, blkk = 0; k < K; k += 32, blkk++) {
            // two adjacent 32x16 (512) block of weight: dst0 & dst1
            auto* dst0 = ret.ptr<ov::bfloat16>(blkn, blkk * 1024);
            auto* dst1 = dst0 + 16 * 32;
            auto valid_k = (K - k) < 32 ? (K - k) : 32;

            auto* src0 = p_weight + n * N_stride + k;
            auto valid_n0 = (N - n) < 16 ? (N - n) : 16;
            repackB<T>(dst0, src0, N_stride, valid_n0, valid_k);

            auto* src1 = p_weight + (n + 16) * N_stride + k;
            auto valid_n1 = (N - (n + 16)) < 16 ? (N - (n + 16)) : 16;
            repackB<T>(dst1, src1, N_stride, valid_n1, valid_k);
        }
    }
}

template void MKernel::prepareB<ov::bfloat16>(PlainTensor& ret, ov::bfloat16* p_weight, int stride, int N, int K);

// run L2 cache blocking kernel with size:
//    [BM, BK]*[BK, BN] => [BM, BN]
//
// prefetch of A can be done inside of this level of kernel
// since it's done in unit of 32-rows
// but prefetch of next B must be specified by caller.
//
void MKernel::run(int M,  // actual M
            uint8_t* pA,
            int strideA,              // A [M, K]
            PlainTensor& repacked_B,  // B [N/32, K*32] ov::bfloat16
            uint8_t* pC,
            int strideC,          // C [M, N]
            uint8_t* prefetch_B,  // prefetch B
            bool do_accumulation) {
    call_args args;
    // number of blocks in N dimension (in unit of 32 columns)
    auto num_blkN = static_cast<int>(repacked_B.size(0));
    auto K = repacked_B.size(1) / 32;
    auto* pB = repacked_B.ptr<uint8_t>();
    auto strideB = repacked_B.stride_bytes(0);

    args.do_accumulation = do_accumulation;
    args.k_tiles = K / 32;
    args.strideA = strideA;
    args.strideC = strideC;
    args.prefetch = prefetch_B;
    assert((K % 32) == 0);

    auto prefetch_step = m_prefetch_Blines * 64 * args.k_tiles;

    // if (BM != m_BM_hint) it only effect prefetch of B which is not vital to function
    for (int m = 0; m < M; m += 32, pA += 32 * strideA, pC += 32 * strideC) {
        args.pB = pB;
        args.M = std::min(M - m, 32);
        args.pA = pA;
        for (int ni = 0; ni < num_blkN; ni++, args.pB += strideB, args.prefetch += prefetch_step) {
            args.pC = pC + ni * 32 * sizeof(float);
            (*this)(&args);
        }
    }
}

void GateUpCombine::generate() {
    Xbyak::Label loop_begin;

    Xbyak::Reg64 src = abi_param1;
    Xbyak::Reg64 dst = abi_param2;
    Xbyak::Reg64 prefetch_dst = abi_param3;
    Xbyak::Reg64 BN = abi_param4;

    Xbyak::Reg64 loop_i = rax;
    const auto zmm_gate = zmm5;
    const auto zmm_silu = zmm6;
    const auto zmm_up = zmm0;
    const auto ymm_dst = ymm5;

    // when save_state is false, push/pop will not be generated.
    auto injector = std::make_shared<jit_uni_eltwise_injector_f32<dnnl::impl::cpu::x64::avx512_core>>(
        this,
        m_act_alg,
        1.f,
        1.0f,
        1.f,
        false,                              // save_state, state will be saved in our function
        Xbyak::Reg64(Xbyak::Operand::R10),  // p_table
        Xbyak::Opmask(1),                   // k_mask
        true,                               // is_fwd
        false,                              // use_dst
        false,                              // preserve_vmm
        false);                             // preserve_p_table

    xor_(loop_i, loop_i);
    injector->load_table_addr();

    shr(BN, 1);  // BN = BN/2;
    align(64);
    L(loop_begin);
    {
        vmovups(zmm_gate, ptr[src + loop_i * 8]);
        // silu will internally use zmm0~zmm3, gelu will use ~zmm4
        vmovups(zmm_silu, zmm_gate);
        injector->compute_vector(zmm_silu.getIdx());
        vmovups(zmm_up, ptr[src + loop_i * 8 + 16 * 4]);
        vmulps(zmm_up, zmm_up, zmm_silu);
        vcvtneps2bf16(ymm_dst, zmm_up);
        prefetchwt1(ptr[prefetch_dst + loop_i * 2]);
        vmovdqu(ptr[dst + loop_i * 2], ymm_dst);
    }
    add(loop_i, 16);
    cmp(loop_i, BN);
    jl(loop_begin, T_NEAR);

    ret();

    injector->prepare_table();
}

void ReduceAdd2bh::generate() {
    if (m_do_reduce2) {
        Xbyak::Reg64 src0 = abi_param1;
        Xbyak::Reg64 src1 = abi_param2;
        Xbyak::Reg64 dst = abi_param3;
        Xbyak::Reg64 prefetch_dst = abi_param4;
        Xbyak::Reg64 BN = abi_param5;
        Xbyak::Reg64 loop_i = rax;

        Xbyak::Label loop_begin;

        xor_(loop_i, loop_i);

        align(64, false);
        L(loop_begin);
        {
            vmovups(zmm0, ptr[src0 + loop_i * 4]);
            vmovups(zmm1, ptr[src1 + loop_i * 4]);
            vmovups(zmm2, ptr[src0 + loop_i * 4 + 16 * 4]);
            vmovups(zmm3, ptr[src1 + loop_i * 4 + 16 * 4]);
            vaddps(zmm0, zmm0, zmm1);
            vaddps(zmm2, zmm2, zmm3);
            vcvtne2ps2bf16(zmm4, zmm2, zmm0);
            prefetchwt1(ptr[prefetch_dst + loop_i * 2]);
            vmovups(ptr[dst + loop_i * 2], zmm4);
        }
        add(loop_i, 32);
        cmp(loop_i, BN);
        jl(loop_begin, T_NEAR);

        ret();
    } else {
        Xbyak::Reg64 src0 = abi_param1;
        Xbyak::Reg64 dst = abi_param2;
        Xbyak::Reg64 prefetch_dst = abi_param3;
        Xbyak::Reg64 BN = abi_param4;
        Xbyak::Reg64 loop_i = rax;

        Xbyak::Label loop_begin;

        xor_(loop_i, loop_i);

        align(64, false);
        L(loop_begin);
        {
            vmovups(zmm0, ptr[src0 + loop_i * 4]);
            vmovups(zmm2, ptr[src0 + loop_i * 4 + 16 * 4]);
            vcvtne2ps2bf16(zmm4, zmm2, zmm0);
            prefetchwt1(ptr[prefetch_dst + loop_i * 2]);
            vmovups(ptr[dst + loop_i * 2], zmm4);
        }
        add(loop_i, 32);
        cmp(loop_i, BN);
        jl(loop_begin, T_NEAR);

        ret();
    }
}

}  // namespace intel_cpu
}  // namespace ov
