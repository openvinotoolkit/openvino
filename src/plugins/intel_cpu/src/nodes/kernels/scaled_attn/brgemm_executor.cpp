// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_executor.hpp"

#include "dnnl_extension_utils.h"
#include "utils/cpu_utils.hpp"

using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64::matmul;
#define THROW_ERROR                                                   \
    IE_THROW() << "oneDNN 1st token executor with name Init Failure'" \
               << "' "

namespace ov {
namespace intel_cpu {
namespace node {

brgemmExecutor::brgemmExecutor(size_t M, size_t K, size_t N, size_t lda, size_t ldb, size_t ldc)
    : M(M),
      K(K),
      N(N),
      lda(lda),
      ldb(ldb),
      ldc(ldc) {
    // blocking M
    const size_t matmulOptimalM = 32;
    M_blk = matmulOptimalM;
    M_tail = M % M_blk;
    ov::element::Type brg0Prc = ov::element::bf16;
    brg0VnniFactor = 4 / brg0Prc.size();

    // blocing N
    N_blk = 32;
    N_tail = N % N_blk;
    // blocing N
    K_blk = 32;
    K_tail = K % K_blk;
    packedBSize = rnd_up(K, brg0VnniFactor) * rnd_up(N, N_blk) * brg0Prc.size();
    packedBData.resize(packedBSize);
    size_t brg0BaseIdx = std::numeric_limits<size_t>::max();
    for (size_t m = 0; m < 2; m++) {
        for (size_t k = 0; k < 2; k++) {
            for (size_t n = 0; n < 2; n++) {
                auto& brgemmCtx = brgCtxs0[getBrgIdx(m, k, n)];

                auto M_ = m ? M_tail : M < M_blk ? 0 : M_blk;
                auto N_ = n ? N_tail : N - N_tail;
                auto K_ = k ? K_tail : K - K_tail;
                auto beta = k && brgCtxs0[getBrgIdx(m, 0, n)].K != 0 ? 1.0f : 0.0f;

                brgemmCtx.M = M_;
                brgemmCtx.N = N_;
                brgemmCtx.K = K_;
                brgemmCtx.LDA = lda;
                brgemmCtx.LDB = rnd_up(N, N_blk);  // ???
                brgemmCtx.LDC = ldc;
                brgemmCtx.dt_in0 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(brg0Prc));
                brgemmCtx.dt_in1 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(brg0Prc));
                brgemmCtx.beta = beta;
                brgemmCtx.is_with_amx = true;

                // don't create brgemm kernels for empty tiles
                if (M_ != 0 && K_ != 0 && N_ != 0) {
                    if (brg0BaseIdx == std::numeric_limits<size_t>::max())
                        brg0BaseIdx = getBrgIdx(m, k, n);
                    init_brgemm(brgemmCtx, brgKernels[getBrgIdx(m, k, n)], true);
                }
            }
        }
    }

    auto& brgemmCtx0 = brgCtxs0[brg0BaseIdx];

    // TODO: matrix A copy should be performed to enable AMX matmuls for arbitrary shapes
    if (brgemmCtx0.is_with_amx && K_tail) {
        init_brgemm_copy_a(brgCopyAKernel, K, K_blk, K_tail, brgemmCtx0.LDA, brgemmCtx0.dt_in0);
    }

    if (brgemmCtx0.is_with_amx || brg0Prc == ov::element::i8 || brg0Prc == ov::element::bf16) {
        init_brgemm_copy_b(brgCopyBKernel,
                           N,
                           N_blk,
                           N_tail,
                           brgemmCtx0.LDB,
                           brgemmCtx0.K,
                           brgemmCtx0.is_with_amx,
                           brgemmCtx0.dt_in0,
                           brgemmCtx0.dt_in1,
                           ldb == 1 ? true : false);
    }
}

void brgemmExecutor::init_brgemm(brgemmCtx& ctx,
                                 std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& brgKernel,
                                 bool use_amx) {
#ifdef OPENVINO_ARCH_X86_64
    brgemm_t brgDesc;

    const bool is_int8 =
        one_of(ctx.dt_in0, data_type::u8, data_type::s8) && one_of(ctx.dt_in1, data_type::u8, data_type::s8);
    auto isa = use_amx                                     ? isa_undef
               : ctx.dt_in0 == dnnl_data_type_t::dnnl_bf16 ? avx512_core_bf16
                                                           : (is_int8 ? avx512_core_vnni : avx512_core);
    auto status = brgemm_desc_init(&brgDesc,
                                   isa,
                                   brgemm_addr,
                                   ctx.dt_in0,
                                   ctx.dt_in1,
                                   ctx.transpose_a,
                                   ctx.transpose_b,
                                   brgemm_row_major,
                                   1.f,
                                   ctx.beta,
                                   ctx.LDA,
                                   ctx.LDB,
                                   ctx.LDC,
                                   ctx.M,
                                   ctx.N,
                                   ctx.K,
                                   nullptr);
    if (status != dnnl_success) {
        THROW_ERROR << "cannot be executed due to invalid brgconv params";
    }

    ctx.is_with_amx = use_amx;
    status = brgemm_init_tiles(brgDesc, ctx.palette);
    if (use_amx) {
        amx_tile_configure(ctx.palette);
    }

    ctx.is_with_comp = ctx.dt_in0 == dnnl_data_type_t::dnnl_s8 && !ctx.is_with_amx;

    brgemm_kernel_t* brgKernel_ = nullptr;
    status = brgemm_kernel_create(&brgKernel_, brgDesc);
    if (status != dnnl_success) {
        THROW_ERROR << "cannot be executed due to invalid brgconv params";
    }
    brgKernel.reset(brgKernel_);
#else
    THROW_ERROR << "is not supported on non-x86_64";
#endif  // OPENVINO_ARCH_X86_64
}
void brgemmExecutor::init_brgemm_copy_a(
    std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_a_t>& brgCopyKernel,
    size_t K,
    size_t K_blk,
    size_t K_tail,
    size_t LDA,
    dnnl_data_type_t dt_in0,
    bool transpose) {
    brgemm_matmul_conf_t brgCopyKernelConf;
    brgCopyKernelConf.src_tag = dnnl_abcd;
    brgCopyKernelConf.K = K;
    brgCopyKernelConf.K_tail = K_tail;
    brgCopyKernelConf.K_blk = K_blk;
    brgCopyKernelConf.use_buffer_a_tail_only = false;
    brgCopyKernelConf.LDA = LDA;
    brgCopyKernelConf.has_zero_point_b = false;
    brgCopyKernelConf.s8s8_compensation_required = false;
    brgCopyKernelConf.wei_zp_type = dnnl::impl::cpu::x64::none;
    brgCopyKernelConf.src_zp_type = dnnl::impl::cpu::x64::none;
    brgCopyKernelConf.src_dt = dt_in0;
    brgCopyKernelConf.a_dt_sz = DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(dt_in0));
    brgCopyKernelConf.transposed_A = transpose;

#if defined(OPENVINO_ARCH_X86_64)
    create_brgemm_matmul_copy_a(brgCopyKernel, &brgCopyKernelConf);
#endif  // OPENVINO_ARCH_X86_64
}

void brgemmExecutor::init_brgemm_copy_b(
    std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t>& brgCopyKernel,
    size_t N,
    size_t N_blk,
    size_t N_tail,
    size_t LDB,
    size_t K,
    bool is_with_amx,
    dnnl_data_type_t dt_in0,
    dnnl_data_type_t dt_in1,
    bool transpose) {
    brgemm_matmul_conf_t brgCopyKernelConf;
    brgCopyKernelConf.src_dt = dt_in0;
    brgCopyKernelConf.wei_dt = dt_in1;
    brgCopyKernelConf.wei_n_blk = N_blk;
    brgCopyKernelConf.wei_tag = transpose ? dnnl_ba : dnnl_ab;
    brgCopyKernelConf.copy_B_wei_stride = 0;
    brgCopyKernelConf.LDB = LDB;
    brgCopyKernelConf.N = N;
    brgCopyKernelConf.N_tail = N_tail;
    brgCopyKernelConf.N_blk = N_blk;
    brgCopyKernelConf.K = K;
    brgCopyKernelConf.K_blk = K;
    brgCopyKernelConf.N_chunk_elems = brgCopyKernelConf.N_blk;
    brgCopyKernelConf.b_dt_sz =
        DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(brgCopyKernelConf.src_dt));
    brgCopyKernelConf.tr_b_dt_sz =
        DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(brgCopyKernelConf.src_dt));
    brgCopyKernelConf.req_wei_vnni_downconvert = false;

    if (is_with_amx) {
        brgCopyKernelConf.isa = avx512_core_amx;
        brgCopyKernelConf.s8s8_compensation_required = false;
    } else {
        brgCopyKernelConf.isa = dt_in0 == dnnl_data_type_t::dnnl_bf16 ? avx512_core_bf16 : avx512_core_vnni;
        brgCopyKernelConf.s8s8_compensation_required = dt_in0 == dnnl_data_type_t::dnnl_s8;
    }

    brgCopyKernelConf.has_zero_point_a = false;
    brgCopyKernelConf.has_zero_point_b = false;
    brgCopyKernelConf.src_zp_type = dnnl::impl::cpu::x64::none;

#if defined(OPENVINO_ARCH_X86_64)
    auto ret = create_brgemm_matmul_copy_b(brgCopyKernel, &brgCopyKernelConf);
    if (ret != dnnl::impl::status_t::dnnl_success)
        THROW_ERROR << "cannot create_brgemm_matmul_copy_b kernel, dnnl_status: ";
#endif  // OPENVINO_ARCH_X86_64
}

void brgemmExecutor::executeGemm(void* a, void* b, void* c) {
    auto ptr_a = reinterpret_cast<uint8_t*>(a);
    auto ptr_b = reinterpret_cast<uint8_t*>(b);
    auto ptr_c = reinterpret_cast<uint8_t*>(c);
    auto dataType = ov::element::bf16;
    if (brgCopyBKernel) {
        for (size_t nb = 0; nb < div_up(N, N_blk); nb++) {
            auto pCopyKernel0In = ptr_b + nb * N_blk * dataType.size();
            auto pCopyKernel0Out = packedBData.data() + nb * N_blk * brg0VnniFactor * dataType.size();

            auto ctx = jit_brgemm_matmul_copy_b_t::ctx_t();

            const bool is_N_tail = (N - nb * N_blk < N_blk);
            ctx.current_N_blk = is_N_tail ? N_tail : N_blk;
            ctx.src = pCopyKernel0In;
            ctx.tr_src = pCopyKernel0Out;
            ctx.compensation_ptr = nullptr;
            ctx.zp_a_compensation_ptr = nullptr;
            ctx.zp_a_neg_value_ptr = nullptr;
            ctx.current_K_start = 0;
            ctx.current_K_iters = K;

            (*brgCopyBKernel)(&ctx);
        }
    }
    size_t brgIdx0 = getBrgIdx(0, 0, 0);
    // The step for matrix A over main K dimension
    size_t K0_step0 = brgCtxs0[brgIdx0].K;
    // The step for matrix B over main K dimension
    size_t K0_step1 = brgCtxs0[brgIdx0].K * brgCtxs0[brgIdx0].LDB;
    // The step for matrix B over N dimension
    size_t N0_step0 = brgCtxs0[brgIdx0].N * brg0VnniFactor;
    // The step for matrix C over N dimension
    size_t N0_step1 = brgCtxs0[brgIdx0].N;
    for (size_t mb = 0; mb < div_up(M, M_blk); mb++) {
        const bool is_M_tail = (M - mb * M_blk < M_blk);
        for (size_t n = 0; n < 2; n++) {
            for (size_t k = 0; k < 2; k++) {
                size_t mIdx = is_M_tail ? 1 : 0;
                auto& brgemmCtx = brgCtxs0[getBrgIdx(mIdx, k, n)];

                if (brgemmCtx.K != 0 && brgemmCtx.N != 0) {
                    callBrgemm(brgemmCtx,
                               brgKernels[getBrgIdx(mIdx, k, n)],
                               ptr_a + (mb * M_blk * brgemmCtx.LDA) * ov::element::bf16.size(),
                               packedBData.data() + (k * K0_step1 + n * N0_step0) * ov::element::bf16.size(),
                               ptr_c + (mb * M_blk * brgemmCtx.LDC + n * N0_step1) * ov::element::bf16.size(),
                               nullptr);
                }
            }
        }
    }
}
void brgemmExecutor::callBrgemm(brgemmCtx& ctx,
                                std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& brgKernel,
                                const void* pin0,
                                const void* pin1,
                                void* pout,
                                void* wsp) {
#if defined(OPENVINO_ARCH_X86_64)
    if (ctx.is_with_amx)
        amx_tile_configure(ctx.palette);
    if (ctx.is_with_comp) {
        brgemm_post_ops_data_t post_ops_data;
        brgemm_kernel_execute_postops(brgKernel.get(), 1, pin0, pin1, nullptr, pout, pout, post_ops_data, wsp);
    } else {
        brgemm_batch_element_t addr_batch;
        addr_batch.ptr.A = pin0;
        addr_batch.ptr.B = pin1;
        brgemm_kernel_execute(brgKernel.get(), 1, &addr_batch, pout, nullptr, nullptr);
    }
#else
    THROW_ERROR("is not supported on non-x64 platforms");
#endif  // OPENVINO_ARCH_X86_64
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov