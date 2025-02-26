// Copyright (C) 2018-2025 Intel Corporation
// Copyright (C) 2024 FUJITSU LIMITED
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_kernel.hpp"

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <openvino/core/except.hpp>

#include "dnnl_extension_utils.h"
#include "utils/cpu_utils.hpp"

using namespace dnnl::impl::cpu::aarch64;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::aarch64::matmul;

#define THROW_ERROR(...) OPENVINO_THROW("brgemm executor Init Failure '", __VA_ARGS__)

namespace ov::intel_cpu {

static size_t getVlen() {
    return mayiuse(sve_512)   ? cpu_isa_traits<sve_512>::vlen
           : mayiuse(sve_256) ? cpu_isa_traits<sve_256>::vlen
                              : cpu_isa_traits<sve_128>::vlen;
}

BrgemmKernel::BrgemmKernel(size_t M,
                           size_t N,
                           size_t K,
                           size_t lda,
                           size_t ldb,
                           size_t ldc,
                           bool b_transposed,
                           ov::element::Type inType,
                           bool b_accumulate)
    : M(M),
      M_blk(matmulOptimalM),
      M_tail(M % M_blk),
      K(K),
      K_blk(K),
      K_tail(K % K_blk),
      N(N),
      N_blk(std::max(N, getVlen() / inType.size())),
      N_tail(N % N_blk),
      lda(lda),
      ldb(ldb),
      ldc(ldc),
      b_transposed(b_transposed),
      kBlkStep(4 / inType.size()),
      packedBSize(rnd_up(K, getVlen() / inType.size()) * rnd_up(N, N_blk) * inType.size()),
      inType(inType) {
    // copied K must be round up by vlen / inType.size(), otherwise copy B kernel may access wrong memory
    size_t brg0BaseIdx = std::numeric_limits<size_t>::max();
    for (size_t m = 0; m < 2; m++) {
        for (size_t k = 0; k < 2; k++) {
            for (size_t n = 0; n < 2; n++) {
                auto& brgemmCtx = brgCtxs[getBrgIdx(m, k, n)];

                auto M_ = m ? M_tail : M < M_blk ? 0 : M_blk;
                auto N_ = n ? N_tail : N - N_tail;
                auto K_ = k ? K_tail : K - K % K_blk;
                auto beta = (b_accumulate || (k && brgCtxs[getBrgIdx(m, 0, n)].K != 0)) ? 1.0f : 0.0f;

                brgemmCtx.M = M_;
                brgemmCtx.N = N_;
                brgemmCtx.K = K_;
                brgemmCtx.LDA = k ? K_blk : lda;
                brgemmCtx.LDB = b_transposed ? rnd_up(N, N_blk) : ldb;  // b_transposed needs copy
                brgemmCtx.LDC = ldc;
                brgemmCtx.dt_in0 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(inType));
                brgemmCtx.dt_in1 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(inType));
                brgemmCtx.beta = beta;

                // don't create brgemm kernels for empty tiles
                if (M_ != 0 && K_ != 0 && N_ != 0) {
                    if (brg0BaseIdx == std::numeric_limits<size_t>::max()) {
                        brg0BaseIdx = getBrgIdx(m, k, n);
                    }
                    init_brgemm(brgemmCtx, brgKernels[getBrgIdx(m, k, n)]);
                }
            }
        }
    }

    auto& brgemmCtx0 = brgCtxs[brg0BaseIdx];
    if (b_transposed) {
        size_t b_stride = 0;
        b_stride = ldb * inType.size();
        // K should use the original K
        init_brgemm_copy_b(brgCopyBKernel,
                           N,
                           N_blk,
                           N_tail,
                           brgemmCtx0.LDB,
                           K,
                           brgemmCtx0.dt_in0,
                           brgemmCtx0.dt_in1,
                           b_transposed,
                           b_stride);
    }
}

const size_t BrgemmKernel::get_scratch_a_size() const {
    return packedASize;
}

const size_t BrgemmKernel::get_scratch_b_size() const {
    return packedBSize;
}

void BrgemmKernel::init_brgemm(brgemmCtx& ctx, std::unique_ptr<dnnl::impl::cpu::aarch64::brgemm_kernel_t>& brgKernel) {
    brgemm_t brgDesc;
    cpu_isa_t isa;
    isa = mayiuse(sve_512) ? cpu_isa_t::sve_512 : mayiuse(sve_256) ? cpu_isa_t::sve_256 : cpu_isa_t::sve_128;
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
        THROW_ERROR("cannot be executed due to invalid brgconv params");
    }

    brgemm_kernel_t* brgKernel_ = nullptr;
    status = brgemm_kernel_create(&brgKernel_, brgDesc);
    if (status != dnnl_success) {
        THROW_ERROR("cannot be executed due to invalid brgconv params");
    }
    brgKernel.reset(brgKernel_);
}
void BrgemmKernel::init_brgemm_copy_a(
    std::unique_ptr<dnnl::impl::cpu::aarch64::matmul::jit_brgemm_matmul_copy_a_t>& brgCopyKernel,
    size_t K,
    size_t K_blk,
    size_t K_tail,
    size_t LDA,
    dnnl_data_type_t dt_in0,
    bool transpose,
    size_t copy_A_src_stride) {
    brgemm_matmul_conf_t brgCopyKernelConf;
    brgCopyKernelConf.src_tag = dnnl_abcd;
    brgCopyKernelConf.K = K;
    brgCopyKernelConf.K_tail = K_tail;
    brgCopyKernelConf.K_blk = K_blk;
    brgCopyKernelConf.use_buffer_a_tail_only = false;
    // padding K tail to K_blk, LDA is the stride for target tensor
    brgCopyKernelConf.LDA = LDA;
    brgCopyKernelConf.has_zero_point_b = false;
    brgCopyKernelConf.s8s8_compensation_required = false;
    brgCopyKernelConf.wei_zp_type = dnnl::impl::cpu::aarch64::none;
    brgCopyKernelConf.src_zp_type = dnnl::impl::cpu::aarch64::none;
    brgCopyKernelConf.src_dt = dt_in0;
    brgCopyKernelConf.copy_A_src_stride = copy_A_src_stride;
    brgCopyKernelConf.a_dt_sz = DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(dt_in0));
    // copied A has the same precision of original
    brgCopyKernelConf.tr_a_dt_sz = DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(dt_in0));
    brgCopyKernelConf.transposed_A = transpose;
    brgCopyKernelConf.isa = mayiuse(sve_512)   ? cpu_isa_t::sve_512
                            : mayiuse(sve_256) ? cpu_isa_t::sve_256
                                               : cpu_isa_t::sve_128;

    create_brgemm_matmul_copy_a(brgCopyKernel, &brgCopyKernelConf);
}

void BrgemmKernel::init_brgemm_copy_b(
    std::unique_ptr<dnnl::impl::cpu::aarch64::matmul::jit_brgemm_matmul_copy_b_t>& brgCopyKernel,
    size_t N,
    size_t N_blk,
    size_t N_tail,
    size_t LDB,
    size_t K,
    dnnl_data_type_t dt_in0,
    dnnl_data_type_t dt_in1,
    bool transpose,
    size_t copy_B_wei_stride) {
    brgemm_matmul_conf_t brgCopyKernelConf;
    brgCopyKernelConf.src_dt = dt_in0;
    brgCopyKernelConf.wei_dt = dt_in1;
    brgCopyKernelConf.wei_n_blk = N_blk;
    brgCopyKernelConf.wei_tag = transpose ? dnnl_ba : dnnl_ab;
    brgCopyKernelConf.copy_B_wei_stride = copy_B_wei_stride;

    // LDB here is for the target tensor, not source tensor
    brgCopyKernelConf.LDB = LDB;
    brgCopyKernelConf.N = N;
    brgCopyKernelConf.N_tail = N_tail;
    brgCopyKernelConf.N_blk = N_blk;
    brgCopyKernelConf.K = K;
    brgCopyKernelConf.K_blk = K;
    brgCopyKernelConf.K_tail = 0;
    brgCopyKernelConf.N_chunk_elems = brgCopyKernelConf.N_blk;
    brgCopyKernelConf.b_dt_sz =
        DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(brgCopyKernelConf.src_dt));
    brgCopyKernelConf.tr_b_dt_sz =
        DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(brgCopyKernelConf.src_dt));
    brgCopyKernelConf.req_wei_vnni_downconvert = false;
    brgCopyKernelConf.isa = mayiuse(sve_512)   ? cpu_isa_t::sve_512
                            : mayiuse(sve_256) ? cpu_isa_t::sve_256
                                               : cpu_isa_t::sve_128;

    brgCopyKernelConf.has_zero_point_a = false;
    brgCopyKernelConf.has_zero_point_b = false;
    brgCopyKernelConf.src_zp_type = dnnl::impl::cpu::aarch64::none;
    auto ret = create_brgemm_matmul_copy_b(brgCopyKernel, &brgCopyKernelConf);
    if (ret != dnnl::impl::status_t::dnnl_success) {
        THROW_ERROR("cannot create_brgemm_matmul_copy_b kernel");
    }
}

void BrgemmKernel::copy_buffer_b(void* b, void* scratch_b) {
    auto ptr_b = reinterpret_cast<uint8_t*>(b);
    auto ptr_scartch_b = reinterpret_cast<uint8_t*>(scratch_b);
    if (brgCopyBKernel) {
        for (size_t nb = 0; nb < div_up(N, N_blk); nb++) {
            auto N_stride = b_transposed ? ldb : 1;
            auto pCopyKernel0In = ptr_b + nb * N_blk * inType.size() * N_stride;
            auto pCopyKernel0Out = ptr_scartch_b + nb * N_blk * kBlkStep * inType.size();

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
}

void BrgemmKernel::executeGemm(bool is_M_tail, void* a, void* b, void* c, void* wsp, void* scratch_a) {
    auto ptr_A = reinterpret_cast<uint8_t*>(a);
    auto ptr_C = reinterpret_cast<uint8_t*>(c);
    auto ptr_scartch_a = reinterpret_cast<uint8_t*>(scratch_a);
    auto ptr_scartch_b = reinterpret_cast<uint8_t*>(b);
    uint8_t* ptr_a_tail = nullptr;

    size_t brgIdx0 = getBrgIdx(0, 0, 0);
    // The step for matrix A over main K dimension
    size_t K0_step0 = brgCtxs[brgIdx0].K;
    auto cur_M_blk = is_M_tail ? M_tail : M_blk;
    if (brgCopyAKernel) {
        // only copy tailed data;
        size_t K_offset = K < K_blk ? 0 : K0_step0 * inType.size();
        auto pCopyKernelIn = ptr_A + K_offset;
        auto pCopyKernelOut = ptr_scartch_a;

        auto ctx = jit_brgemm_matmul_copy_a_t::ctx_t();

        ctx.current_M_blk = cur_M_blk;
        ctx.zp_b_compensation_buffer_ptr = nullptr;
        ctx.zp_a_compensation_result_ptr = nullptr;
        ctx.zp_b_neg_value_ptr = nullptr;
        ctx.zp_ab_comp_ptr = nullptr;
        ctx.src = pCopyKernelIn;
        ctx.tr_src = pCopyKernelOut;
        ctx.current_K_start = 0;
        ctx.current_K_blk = K % K_blk;

        (*brgCopyAKernel)(&ctx);

        ptr_a_tail = pCopyKernelOut;
    }
    size_t count_N = 0;
    for (size_t n = 0; n < 2; n++) {
        size_t count_K = 0;
        for (size_t k = 0; k < 2; k++) {
            size_t mIdx = is_M_tail ? 1 : 0;
            auto& brgemmCtx = brgCtxs[getBrgIdx(mIdx, k, n)];
            if (brgemmCtx.K != 0 && brgemmCtx.N != 0 && brgemmCtx.M != 0) {
                auto local_a_ptr = k > 0 ? ptr_a_tail : ptr_A;
                auto B_stride = (k * count_K + n * count_N * kBlkStep) * inType.size();
                auto weight_ptr = ptr_scartch_b + B_stride;
                auto C_stride = n * count_N * ov::element::f32.size();
                auto out_ptr = ptr_C + C_stride;
                callBrgemm(brgemmCtx, brgKernels[getBrgIdx(mIdx, k, n)], local_a_ptr, weight_ptr, out_ptr, wsp);
                // stride K, N if body kernel is executed.
                if (k == 0) {
                    count_K = brgemmCtx.K * brgemmCtx.LDB;
                }
                if (n == 0) {
                    count_N = brgemmCtx.N;
                }
            }
        }
    }
}

void BrgemmKernel::executeGemm(void* a, void* b, void* c, void* wsp, void* scratch_a, void* scratch_b) {
    auto ptr_A = reinterpret_cast<uint8_t*>(a);
    auto ptr_B = reinterpret_cast<uint8_t*>(b);
    auto ptr_C = reinterpret_cast<uint8_t*>(c);

    copy_buffer_b(ptr_B, scratch_b);

    for (size_t mb = 0; mb < div_up(M, M_blk); mb++) {
        const bool is_M_tail = (M - mb * M_blk < M_blk);
        auto ptr_a = ptr_A + (mb * M_blk * lda) * inType.size();
        auto ptr_c = ptr_C + (mb * M_blk * ldc) * ov::element::f32.size();
        executeGemm(is_M_tail, ptr_a, scratch_b, wsp, ptr_c, scratch_a);
    }
}
void BrgemmKernel::callBrgemm(brgemmCtx& ctx,
                              std::unique_ptr<dnnl::impl::cpu::aarch64::brgemm_kernel_t>& brgKernel,
                              const void* pin0,
                              const void* pin1,
                              void* pout,
                              void* wsp) {
    brgemm_batch_element_t addr_batch;
    addr_batch.ptr.A = pin0;
    addr_batch.ptr.B = pin1;
    brgemm_kernel_execute(brgKernel.get(), 1, &addr_batch, pout, wsp);
}

}  // namespace ov::intel_cpu
