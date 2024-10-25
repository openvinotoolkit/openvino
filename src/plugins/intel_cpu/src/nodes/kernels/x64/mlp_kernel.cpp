// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlp_kernel.hpp"
#include "emitters/plugin/x64/jit_dnnl_emitters.hpp"
#include "mlp_utils.hpp"
#include "openvino/core/parallel.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu::x64;

using TileConfig = ov::Extensions::Cpu::TileConfig;
using TileConfiger = ov::Extensions::Cpu::TileConfiger;

namespace ov {
namespace intel_cpu {

void MKernel::generate_2x2() {
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

        tmul(tmmC00, tmmA0, tmmB0);
        if (cur_PFB < num_PFB) {
            prefetcht2(ptr[reg_prefetch + cur_PFB * 64]);
            cur_PFB++;
        }

        tileloadd(tmmA1, ptr[reg_A1_addr + reg_A_stride]);
        tmul(tmmC10, tmmA1, tmmB0);
        if (cur_PFB < num_PFB) {
            prefetcht2(ptr[reg_prefetch + cur_PFB * 64]);
            cur_PFB++;
        }

        tileloadd(tmmB1, ptr[reg_B_addr + reg_B_stride]);
        tmul(tmmC01, tmmA0, tmmB1);
        if (cur_PFB < num_PFB) {
            prefetcht2(ptr[reg_prefetch + cur_PFB * 64]);
            cur_PFB++;
        }

        tmul(tmmC11, tmmA1, tmmB1);
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

void MKernel::generate_1x2() {
    Xbyak::Reg64 reg_A_addr = abi_param2;
    Xbyak::Reg64 reg_A_stride = abi_param3;
    Xbyak::Reg64 reg_B_addr = abi_param4;
    Xbyak::Reg64 reg_C_addr = abi_param5;
    Xbyak::Reg64 reg_C_stride = abi_param6;

    Xbyak::Reg64 reg_ktiles = rax;
    Xbyak::Reg64 reg_B_stride = r10;
    // Xbyak::Reg64 reg_prefetch = r12;

    Xbyak::Tmm tmmC00 = tmm0;
    Xbyak::Tmm tmmC01 = tmm1;
    Xbyak::Tmm tmmA0 = tmm4;
    Xbyak::Tmm tmmB0 = tmm6;
    Xbyak::Tmm tmmB1 = tmm7;

    Xbyak::Label loop_over_ktiles;
    Xbyak::Label skip_load;

    {
        auto reg_tmp = reg_B_stride;

        mov(reg_A_addr, ptr[abi_param1 + offsetof(call_args, pA)]);
        mov(reg_A_stride, ptr[abi_param1 + offsetof(call_args, strideA)]);
        mov(reg_B_addr, ptr[abi_param1 + offsetof(call_args, pB)]);
        mov(reg_C_addr, ptr[abi_param1 + offsetof(call_args, pC)]);
        mov(reg_C_stride, ptr[abi_param1 + offsetof(call_args, strideC)]);
        mov(reg_ktiles, ptr[abi_param1 + offsetof(call_args, k_tiles)]);

        mov(reg_tmp, ptr[abi_param1 + offsetof(call_args, do_accumulation)]);
        // new: bit0: 0-skip load from mem, 1-load from mem; bit1: 0-skip tilezero, 1-tilezero; bit2: 0-skip store, 1-store
        mov(abi_param1, reg_tmp);
        and_(reg_tmp, 1);
        jz(skip_load);
        {
            tileloadd(tmmC00, ptr[reg_C_addr + reg_C_stride]);
            tileloadd(tmmC01, ptr[reg_C_addr + reg_C_stride + 64]);
        }
        L(skip_load);
        mov(reg_tmp, abi_param1);
        and_(reg_tmp, 2);
        Xbyak::Label skip_zero;
        jz(skip_zero);
        {
            tilezero(tmmC00);
            tilezero(tmmC01);
        }
        L(skip_zero);
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

        tmul(tmmC00, tmmA0, tmmB0);

        tileloadd(tmmB1, ptr[reg_B_addr + reg_B_stride]);
        tmul(tmmC01, tmmA0, tmmB1);

        lea(reg_A_addr, ptr[reg_A_addr + const_A_steps]);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);
    }
    dec(reg_ktiles);
    jnz(loop_over_ktiles, T_NEAR);

    and_(abi_param1, 4);
    Xbyak::Label skip_store;
    jz(skip_store);
    {
        tilestored(ptr[reg_C_addr + reg_C_stride], tmmC00);
        tilestored(ptr[reg_C_addr + reg_C_stride + 64], tmmC01);
    }
    L(skip_store);

    ret();
}

class FP16ToBF16Kernel : public dnnl::impl::cpu::x64::jit_generator {
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(FP16ToBF16Kernel)
    FP16ToBF16Kernel() : jit_generator("FP16ToBF16Kernel") {
        create_kernel();
    }

    void generate() override {
        Xbyak::Label loop_begin;
        Xbyak::Reg64 src = abi_param1;
        for (int i = 0; i < 16; i++) {
            vcvtph2ps(zmm0, ptr[src]);
            vcvtph2ps(zmm1, ptr[src + 32]);
            vcvtne2ps2bf16(zmm2, zmm1, zmm0);
            vmovups(ptr[src], zmm2);
            lea(src, ptr[src + 64]);
        }

        ret();
    }
};

template <typename Tdst>
static typename std::enable_if<std::is_same<ov::bfloat16, Tdst>::value || std::is_same<ov::float16, Tdst>::value>::type
repackB(Tdst* dst, ov::float16* src, int N_stride, int N, int K) {
    static FP16ToBF16Kernel fp16_to_bf16;
    if (N == 16 && K == 32) {
        // SIMD optimized version
        ov::Extensions::Cpu::XARCH::llm_mlp_transpose_epi32_16x16(dst, src, N_stride * sizeof(Tdst));
        if (std::is_same<ov::bfloat16, Tdst>::value)
            fp16_to_bf16(dst);
        return;
    }

    assert(K <= 32);
    assert(N <= 16);
    int k = 0;
    Tdst zero(0.0f);
    for (; k < 32; k += 2) {
        int n = 0;
        bool is_k0_valid = (k) < K;
        bool is_k1_valid = (k + 1) < K;
        auto* psrc = src + k;
        for (; n < 16 && n < N; n++, psrc += N_stride) {
            *dst++ = is_k0_valid ? static_cast<Tdst>(psrc[0]) : zero;
            *dst++ = is_k1_valid ? static_cast<Tdst>(psrc[1]) : zero;
        }
        for (; n < 16; n++) {
            *dst++ = 0;
            *dst++ = 0;
        }
    }
}

static void repackB(int8_t* dst, int8_t* src, int N_stride, int N, int K) {
    if (N == 16 && K == 64) {
        // SIMD optimized version
        ov::Extensions::Cpu::XARCH::llm_mlp_transpose_epi32_16x16(dst, src, N_stride * sizeof(int8_t));
        return;
    }

    assert(K <= 64);
    assert(N <= 16);
    for (int k = 0; k < 64; k += 4) {
        bool is_k0_valid = (k) < K;
        bool is_k1_valid = (k + 1) < K;
        bool is_k2_valid = (k + 2) < K;
        bool is_k3_valid = (k + 3) < K;
        auto* psrc = src + k;
        int n = 0;
        for (; n < 16 && n < N; n++, psrc += N_stride) {
            *dst++ = is_k0_valid ? psrc[0] : 0;
            *dst++ = is_k1_valid ? psrc[1] : 0;
            *dst++ = is_k2_valid ? psrc[2] : 0;
            *dst++ = is_k3_valid ? psrc[3] : 0;
        }
        for (; n < 16; n++) {
            *dst++ = 0;
            *dst++ = 0;
            *dst++ = 0;
            *dst++ = 0;
        }
    }
}

template<typename Tdst>
void MKernel::BMatrix::setup(Tdst* ext_buff, ov::float16* p_weight, int weight_stride_in_bytes, int N, int K) {
    OPENVINO_ASSERT((N % 32) == 0);
    OPENVINO_ASSERT((K % 32) == 0);

    this->ptr = reinterpret_cast<uint8_t*>(ext_buff);
    this->Bpair_rows = K/32;
    this->Bpair_cols = N/32;

    const int k_step = 32;
    auto N_stride = weight_stride_in_bytes / sizeof(Tdst);
    auto* pdst = reinterpret_cast<int8_t*>(ext_buff);
    for (int n = 0; n < N; n += 32) {
        auto* src0 = p_weight + n * N_stride;
        auto* src1 = p_weight + (n + 16) * N_stride;
        auto valid_n0 = std::min((N - n), 16);
        auto valid_n1 = std::min((N - (n + 16)), 16);
        for (int k = 0, blkk = 0; k < K; k += k_step, blkk++) {
            auto valid_k = std::min((K - k), k_step);
            repackB(reinterpret_cast<Tdst*>(pdst), src0 + k, N_stride, valid_n0, valid_k);
            pdst += 1024;
            repackB(reinterpret_cast<Tdst*>(pdst), src1 + k, N_stride, valid_n1, valid_k);
            pdst += 1024;
        }
    }
}

template void MKernel::BMatrix::setup<ov::bfloat16>(ov::bfloat16*, ov::float16*, int, int, int);
template void MKernel::BMatrix::setup<ov::float16>(ov::float16*, ov::float16*, int, int, int);

void MKernel::BMatrix::setup(int8_t* ext_buff, int8_t* p_weight, int weight_stride_in_bytes, int N, int K) {
    OPENVINO_ASSERT((N % 32) == 0);
    OPENVINO_ASSERT((K % 64) == 0);

    this->ptr = reinterpret_cast<uint8_t*>(ext_buff);
    this->Bpair_rows = K/64;
    this->Bpair_cols = N/32;

    const int k_step = 64;
    auto N_stride = weight_stride_in_bytes / sizeof(int8_t);
    auto* pdst = reinterpret_cast<int8_t*>(ext_buff);
    for (int n = 0; n < N; n += 32) {
        auto* src0 = p_weight + n * N_stride;
        auto* src1 = p_weight + (n + 16) * N_stride;
        auto valid_n0 = std::min((N - n), 16);
        auto valid_n1 = std::min((N - (n + 16)), 16);
        for (int k = 0, blkk = 0; k < K; k += k_step, blkk++) {
            auto valid_k = std::min((K - k), k_step);
            repackB(reinterpret_cast<int8_t*>(pdst), src0 + k, N_stride, valid_n0, valid_k);
            pdst += 1024;
            repackB(reinterpret_cast<int8_t*>(pdst), src1 + k, N_stride, valid_n1, valid_k);
            pdst += 1024;
        }
    }
}

// interleaving two weights into one in unit of 16-column
template<typename Tdst>
void MKernel::BMatrix::setup(Tdst* ext_buff,
                             ov::float16* p_weight_B0,
                             ov::float16* p_weight_B1,
                             int weight_stride_in_bytes,
                             int N,
                             int K) {
    OPENVINO_ASSERT((N % 32) == 0);
    OPENVINO_ASSERT((K % 32) == 0);

    this->ptr = reinterpret_cast<uint8_t*>(ext_buff);
    this->Bpair_rows = K / 32;
    this->Bpair_cols = N / 32;

    const int k_step = 32;
    auto N_stride = weight_stride_in_bytes / sizeof(Tdst);
    auto N2 = N / 2;
    auto* pdst = reinterpret_cast<int8_t*>(ext_buff);
    for (int n = 0; n < N2; n += 16) {
        auto valid_n0 = std::min((N2 - n), 16);
        for (int k = 0; k < K; k += k_step) {
            auto valid_k = std::min((K - k), k_step);
            repackB(reinterpret_cast<Tdst*>(pdst), p_weight_B0 + n * N_stride + k, N_stride, valid_n0, valid_k);
            pdst += 1024;
            repackB(reinterpret_cast<Tdst*>(pdst), p_weight_B1 + n * N_stride + k, N_stride, valid_n0, valid_k);
            pdst += 1024;
        }
    }
}
template void MKernel::BMatrix::setup<ov::bfloat16>(ov::bfloat16*, ov::float16*, ov::float16*, int, int, int);
template void MKernel::BMatrix::setup<ov::float16>(ov::float16*, ov::float16*, ov::float16*, int, int, int);

void MKernel::BMatrix::setup(int8_t* ext_buff,
                             int8_t* p_weight_B0,
                             int8_t* p_weight_B1,
                             int weight_stride_in_bytes,
                             int N,
                             int K) {
    OPENVINO_ASSERT((N % 32) == 0);
    OPENVINO_ASSERT((K % 64) == 0);

    this->ptr = reinterpret_cast<uint8_t*>(ext_buff);
    this->Bpair_rows = K / 64;
    this->Bpair_cols = N / 32;

    const int k_step = 64;
    auto N_stride = weight_stride_in_bytes / sizeof(int8_t);
    auto N2 = N / 2;
    auto* pdst = reinterpret_cast<int8_t*>(ext_buff);
    for (int n = 0; n < N2; n += 16) {
        auto valid_n0 = std::min((N2 - n), 16);
        for (int k = 0; k < K; k += k_step) {
            auto valid_k = std::min((K - k), k_step);
            repackB(reinterpret_cast<int8_t*>(pdst), p_weight_B0 + n * N_stride + k, N_stride, valid_n0, valid_k);
            pdst += 1024;
            repackB(reinterpret_cast<int8_t*>(pdst), p_weight_B1 + n * N_stride + k, N_stride, valid_n0, valid_k);
            pdst += 1024;
        }
    }
}

// run L2 cache blocking kernel with size:
//    [BM, BK]*[BK, BN] => [BM, BN]
//
// prefetch of A can be done inside of this level of kernel
// since it's done in unit of 32-rows
// but prefetch of next B must be specified by caller.
//
void MKernel::run(int M,  // actual M
                  uint8_t* pA,
                  int strideA,          // A [M, K]
                  BMatrix& repacked_B,  // B [N/32, K*32] ov::bfloat16
                  uint8_t* pC,
                  int strideC,          // C [M, N]
                  uint8_t* prefetch_B,  // prefetch B
                  bool do_accumulation) {
    call_args args;

    auto* pB = repacked_B.ptr;
    auto strideB = repacked_B.Bpair_rows * repacked_B.Bpair_size;
    auto num_blkN = repacked_B.Bpair_cols;

    args.do_accumulation = do_accumulation;
    args.k_tiles = repacked_B.Bpair_rows;
    args.strideA = strideA;
    args.strideC = strideC;
    args.prefetch = prefetch_B;

    auto prefetch_step = m_prefetch_Blines * 64 * args.k_tiles;

    // if (BM != m_BM_hint) it only effect prefetch of B which is not vital to function
    for (int m = 0; m < M; m += 32, pA += 32 * strideA, pC += 32 * strideC) {
        args.pB = pB;
        args.M = std::min(M - m, 32);
        args.pA = pA;
        for (size_t ni = 0; ni < num_blkN; ni++, args.pB += strideB, args.prefetch += prefetch_step) {
            args.pC = pC + ni * 32 * sizeof(float);
            (*this)(&args);
        }
    }
}

void MatrixDynQuantPerRow::quantize(size_t BM, ov::bfloat16* psrc, int src_stride) {
    assert(BM <= M);
    parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
        size_t start{0}, end{0};
        splitter(BM, nthr, ithr, start, end);
        ov::Extensions::Cpu::XARCH::llm_mlp_quantize_bf16_i8(psrc + start * src_stride,
                                                            src_stride,
                                                            data + start * K,
                                                            K,
                                                            end - start,
                                                            K,
                                                            scale + start,
                                                            zp + start,
                                                            asym);
    });
}

void MatrixDynQuantPerRow::quantize(size_t BM, ov::float16* psrc, int src_stride) {
    assert(BM <= M);
    parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
        size_t start{0}, end{0};
        splitter(BM, nthr, ithr, start, end);
        ov::Extensions::Cpu::XARCH::llm_mlp_quantize_f16_i8(psrc + start * src_stride,
                                                            src_stride,
                                                            data + start * K,
                                                            K,
                                                            end - start,
                                                            K,
                                                            scale + start,
                                                            zp + start,
                                                            asym);
    });
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

    auto injector = std::make_shared<jit_uni_eltwise_injector_f32<dnnl::impl::cpu::x64::avx512_core>>(
        this,
        m_act_alg,
        1.f,
        1.0f,
        1.f,
        true,                               // save_state, true due to additional r15 is used.
        Xbyak::Reg64(Xbyak::Operand::R10),  // p_table
        Xbyak::Opmask(1),                   // k_mask
        true,                               // is_fwd
        false,                              // use_dst
        false,                              // preserve_vmm
        false);                             // preserve_p_table, false due to it will be saved in the function

    push(r10);
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
        if (m_to_f16) {
            vcvtps2ph(ymm_dst, zmm_up, 0x4);
        } else {
            vcvtneps2bf16(ymm_dst, zmm_up);
        }
        prefetchwt1(ptr[prefetch_dst + loop_i * 2]);
        vmovdqu(ptr[dst + loop_i * 2], ymm_dst);
    }
    add(loop_i, 16);
    cmp(loop_i, BN);
    jl(loop_begin, T_NEAR);

    pop(r10);
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
            if (m_to_f16) {
                vcvtps2ph(ptr[dst + loop_i * 2], zmm0, 0x4);
                vcvtps2ph(ptr[dst + loop_i * 2 + 32], zmm2, 0x4);
                prefetchwt1(ptr[prefetch_dst + loop_i * 2]);
           } else {
                vcvtne2ps2bf16(zmm4, zmm2, zmm0);
                prefetchwt1(ptr[prefetch_dst + loop_i * 2]);
                vmovups(ptr[dst + loop_i * 2], zmm4);
            }
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
            if (m_to_f16) {
                vcvtps2ph(ptr[dst + loop_i * 2], zmm0, 0x4);
                vcvtps2ph(ptr[dst + loop_i * 2 + 32], zmm2, 0x4);
                prefetchwt1(ptr[prefetch_dst + loop_i * 2]);
            } else {
                vcvtne2ps2bf16(zmm4, zmm2, zmm0);
                prefetchwt1(ptr[prefetch_dst + loop_i * 2]);
                vmovups(ptr[dst + loop_i * 2], zmm4);
            }
        }
        add(loop_i, 32);
        cmp(loop_i, BN);
        jl(loop_begin, T_NEAR);

        ret();
    }
}

}  // namespace intel_cpu
}  // namespace ov
