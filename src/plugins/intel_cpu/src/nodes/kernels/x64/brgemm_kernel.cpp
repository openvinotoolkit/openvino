// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_kernel.hpp"

#include "dnnl_extension_utils.h"
#include "utils/cpu_utils.hpp"
#include <cpu/x64/cpu_isa_traits.hpp>
#include <openvino/core/except.hpp>

using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64::matmul;

#define THROW_ERROR(...) OPENVINO_THROW("brgemm executor Init Failure '", __VA_ARGS__)
namespace ov {
namespace intel_cpu {

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
      K(K),
      N(N),
      lda(lda),
      ldb(ldb),
      ldc(ldc),
      b_transposed(b_transposed),
      inType(inType),
      b_accumulate(b_accumulate) {
    // blocking M
    M_blk = matmulOptimalM;
    M_tail = M % M_blk;

    if (!one_of(inType, ov::element::bf16, ov::element::f16, ov::element::f32))
        THROW_ERROR("brgemm kernel only supports f16, bf16, f32");
    bool is_f32 = inType == ov::element::f32;

    bool is_bf16 = inType == ov::element::bf16;
    if (is_bf16 && !mayiuse(avx512_core_bf16))
        THROW_ERROR("brgemm bf16 kernel could only be used above avx512_bf16");

    bool is_f16 = inType == ov::element::f16;
    if (is_f16 && !mayiuse(avx512_core_fp16))
        THROW_ERROR("brgemm f16 kernel could only be used above avx512_f16");

    srcType = weiType = inType;
    // If isa is avx512_core_fp16, f16 is supported by upconverted to f32
    is_avx_f16_only = inType == ov::element::f16 && mayiuse(avx512_core_fp16) && !mayiuse(avx512_core_amx_fp16);
    if (is_avx_f16_only) {
        srcType = ov::element::f32;
        weiType = ov::element::f32;
    }
    brgVnniFactor = 4 / weiType.size();

    /*
                AVX    AMX
        fp32     Y      N
        bf16     Y      Y
        fp16     Y      Y
    */
    bool isAMXSupported = (is_bf16 && mayiuse(avx512_core_amx)) || (is_f16 && mayiuse(avx512_core_amx_fp16));
    bool isBrgWithAMX = isAMXSupported && !is_avx_f16_only;

    size_t vlen;
    if (mayiuse(avx512_core))
        vlen = cpu_isa_traits<avx512_core>::vlen;
    else
        vlen = cpu_isa_traits<cpu_isa_t::avx2>::vlen;
    // blocking N
    N_blk = !is_f32 ? 32 : std::max(N, vlen / inType.size());
    N_tail = N % N_blk;

    // blocking K
    K_blk = isBrgWithAMX ? 32 : K;
    K_tail = K % K_blk;
    if (isBrgWithAMX && K_tail) {
        K_tail = rnd_up(K_tail, 2);
    }
    // copied K must be round up by vlen / inType.size(), otherwise copy B kernel may access wrong memory
    packedBSize = rnd_up(K, vlen / weiType.size()) * rnd_up(N, N_blk) * weiType.size();
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
                brgemmCtx.LDA = k ? K_blk : (is_avx_f16_only ? K : lda); // f16 use f32 internally
                brgemmCtx.LDB = (!is_f32 || b_transposed) ? rnd_up(N, N_blk) : ldb;  // bf16/fp16/b_transposed needs copy
                brgemmCtx.LDC = ldc;
                brgemmCtx.dt_in0 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(srcType));
                brgemmCtx.dt_in1 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::ElementTypeToDataType(weiType));
                brgemmCtx.beta = beta;

                // don't create brgemm kernels for empty tiles
                if (M_ != 0 && K_ != 0 && N_ != 0) {
                    if (brg0BaseIdx == std::numeric_limits<size_t>::max())
                        brg0BaseIdx = getBrgIdx(m, k, n);
                    init_brgemm(brgemmCtx, brgKernels[getBrgIdx(m, k, n)], isBrgWithAMX);
                }
            }
        }
    }

    auto& brgemmCtx0 = brgCtxs[brg0BaseIdx];

    if ((brgemmCtx0.is_with_amx && K_tail) || is_avx_f16_only) {
        init_brgemm_copy_a(brgCopyAKernel,
                           K,
                           K_blk,
                           K_tail,
                           is_avx_f16_only ? K : K_blk,
                           brgemmCtx0.dt_in0,
                           false,
                           lda * inType.size());
        packedASize = M_blk * rnd_up(K, brgemmCtx0.LDA) * srcType.size();
    }

    if (brgemmCtx0.is_with_amx || !is_f32 || b_transposed) {
        size_t b_stride = 0;
        b_stride = ldb * inType.size();
        // K should use the original K
        init_brgemm_copy_b(brgCopyBKernel,
                           N,
                           N_blk,
                           N_tail,
                           brgemmCtx0.LDB,
                           K,
                           brgemmCtx0.is_with_amx,
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

void BrgemmKernel::init_brgemm(brgemmCtx& ctx,
                                 std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& brgKernel,
                                 bool use_amx) {
    brgemm_desc_t brgDesc;

    const bool is_int8 =
        one_of(ctx.dt_in0, data_type::u8, data_type::s8) && one_of(ctx.dt_in1, data_type::u8, data_type::s8);
    cpu_isa_t isa;
    if (use_amx) {
        isa = isa_undef;
    } else if (mayiuse(avx512_core)) {
        if (ctx.dt_in0 == dnnl_data_type_t::dnnl_bf16 && mayiuse(avx512_core_bf16)) {
            isa = avx512_core_bf16;
        } else if (ctx.dt_in0 == dnnl_data_type_t::dnnl_f16 && mayiuse(avx512_core_fp16)) {
            isa = avx512_core_fp16;
        } else {
            if (is_int8) {
                isa = avx512_core_vnni;
            } else {
                isa = avx512_core;
            }
        }
    } else {
        isa = cpu_isa_t::avx2;
    }
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
        THROW_ERROR("cannot be executed due to invalid brgemm params");
    }

    if (use_amx && b_accumulate) {
        brgemm_attr_t brgattr;
        brgattr.max_bs = 1;
        brgattr.wary_tail_read = false;
        brgattr.hint_innermost_loop = brgemm_innermost_undef;
        // if b_accumulate is true, it means we want c+=a*b. jit_brgemm_amx_uker_base_t::load_accumulators can support this using tileload(c) without postops
        brgattr.use_uker = true;
        brgattr.use_interleave_stores = true;
        brgattr.hint_prefetching = brgemm_kernel_prefetching_t::brgemm_prf1;
        if (brgemm_desc_set_attr(&brgDesc, brgattr) != dnnl_success) {
            THROW_ERROR("cannot be executed due to brgemm_desc_set_attr failed");
        }
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
        THROW_ERROR("cannot be executed due to invalid brgconv params");
    }
    brgKernel.reset(brgKernel_);
}

void BrgemmKernel::init_brgemm_copy_a(
    std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_a_t>& brgCopyKernel,
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
    //padding K tail to K_blk, LDA is the stride for target tensor
    brgCopyKernelConf.LDA = LDA;
    brgCopyKernelConf.has_zero_point_b = false;
    brgCopyKernelConf.s8s8_compensation_required = false;
    brgCopyKernelConf.wei_zp_type = dnnl::impl::cpu::x64::none;
    brgCopyKernelConf.src_zp_type = dnnl::impl::cpu::x64::none;
    brgCopyKernelConf.src_dt = is_avx_f16_only ? dnnl_data_type_t::dnnl_f32 : dt_in0;
    brgCopyKernelConf.copy_A_src_stride = copy_A_src_stride;
    // copy_a_kernel assumes that in/out tensor has same data type except f16
    // copy_a_kernel has special path for f16: assuming input(f16) -> output(f32)
    brgCopyKernelConf.a_dt_sz = is_avx_f16_only ? sizeof(ov::float16) : DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(dt_in0));
    // copied A has the same precision of original
    brgCopyKernelConf.tr_a_dt_sz = is_avx_f16_only ? sizeof(float) : DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(dt_in0));
    brgCopyKernelConf.transposed_A = transpose;
    brgCopyKernelConf.isa = is_avx_f16_only ? avx512_core_fp16 : avx512_core_amx;

    create_brgemm_matmul_copy_a(brgCopyKernel, &brgCopyKernelConf);
}

void BrgemmKernel::init_brgemm_copy_b(
    std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t>& brgCopyKernel,
    size_t N,
    size_t N_blk,
    size_t N_tail,
    size_t LDB,
    size_t K,
    bool is_with_amx,
    dnnl_data_type_t dt_in0,
    dnnl_data_type_t dt_in1,
    bool transpose,
    size_t copy_B_wei_stride) {
    brgemm_matmul_conf_t brgCopyKernelConf;
    brgCopyKernelConf.src_dt = is_avx_f16_only ? dnnl_data_type_t::dnnl_f32 : dt_in0;
    brgCopyKernelConf.wei_dt = is_avx_f16_only ? dnnl_data_type_t::dnnl_f32 : dt_in1;
    brgCopyKernelConf.orig_wei_dt = dt_in1;
    brgCopyKernelConf.wei_n_blk = N_blk;
    brgCopyKernelConf.wei_tag =  transpose ? dnnl_ba : dnnl_ab;
    brgCopyKernelConf.copy_B_wei_stride = copy_B_wei_stride;
    brgCopyKernelConf.transposed_B = transpose;

    // LDB here is for the target tensor, not source tensor
    brgCopyKernelConf.LDB = LDB;
    brgCopyKernelConf.N = N;
    brgCopyKernelConf.N_tail = N_tail;
    brgCopyKernelConf.N_blk = N_blk;
    brgCopyKernelConf.K = K;
    brgCopyKernelConf.K_blk = K;
    brgCopyKernelConf.K_tail = 0;
    brgCopyKernelConf.N_chunk_elems = brgCopyKernelConf.N_blk;
    // f16 is computed by upconverting. in(f16) -> out(f32)
    brgCopyKernelConf.b_dt_sz = is_avx_f16_only ? sizeof(ov::float16) :
        DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(brgCopyKernelConf.src_dt));
    brgCopyKernelConf.tr_b_dt_sz = is_avx_f16_only ?  sizeof(float) :
        DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(brgCopyKernelConf.src_dt));
    brgCopyKernelConf.req_wei_vnni_downconvert = false;

    if (is_with_amx) {
        brgCopyKernelConf.isa = dt_in0 == dnnl_data_type_t::dnnl_f16 ? avx512_core_amx_fp16 : avx512_core_amx;
        brgCopyKernelConf.s8s8_compensation_required = false;
    } else {
        if (inType == ov::element::f16) {
            brgCopyKernelConf.isa = mayiuse(avx512_core_fp16) ? avx512_core_fp16 : avx2_vnni_2;
        } else {
            brgCopyKernelConf.isa = dt_in0 == dnnl_data_type_t::dnnl_bf16 ? avx512_core_bf16 : avx512_core_vnni;
        }
        brgCopyKernelConf.s8s8_compensation_required = false;
    }

    brgCopyKernelConf.has_zero_point_a = false;
    brgCopyKernelConf.has_zero_point_b = false;
    brgCopyKernelConf.src_zp_type = dnnl::impl::cpu::x64::none;
    auto ret = create_brgemm_matmul_copy_b(brgCopyKernel, &brgCopyKernelConf);
    if (ret != dnnl::impl::status_t::dnnl_success)
        THROW_ERROR("cannot create_brgemm_matmul_copy_b kernel");
}

void BrgemmKernel::copy_buffer_b(void* b, void* scratch_b) {
    auto ptr_b = reinterpret_cast<uint8_t*>(b);
    auto ptr_scartch_b = reinterpret_cast<uint8_t*>(scratch_b);
    if (brgCopyBKernel) {
        for (size_t nb = 0; nb < div_up(N, N_blk); nb++) {
            auto N_stride = b_transposed ? ldb : 1;
            auto pCopyKernel0In = ptr_b + nb * N_blk * inType.size() * N_stride;
            auto pCopyKernel0Out = ptr_scartch_b + nb * N_blk * brgVnniFactor * weiType.size();

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

    size_t brgIdx0 = getBrgIdx(0, 0, 0);
    // The step for matrix A over main K dimension
    size_t K0_step0 = brgCtxs[brgIdx0].K;
    auto cur_M_blk = is_M_tail ? M_tail : M_blk;
    if (brgCopyAKernel) {
        size_t K_offset = is_avx_f16_only ? 0 : (K < K_blk ? 0 : K0_step0 * srcType.size());
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
    }
    size_t count_N = 0;
    for (size_t n = 0; n < 2; n++) {
        size_t count_K = 0;
        for (size_t k = 0; k < 2; k++) {
            size_t mIdx = is_M_tail ? 1 : 0;
            auto& brgemmCtx = brgCtxs[getBrgIdx(mIdx, k, n)];
            if (brgemmCtx.K != 0 && brgemmCtx.N != 0 && brgemmCtx.M != 0) {
                auto local_a_ptr = is_avx_f16_only ? ptr_scartch_a : (k > 0 ? ptr_scartch_a : ptr_A);
                auto B_stride = (k * count_K + n * count_N * brgVnniFactor) * weiType.size();
                auto weight_ptr = ptr_scartch_b + B_stride;
                auto C_stride = n * count_N * ov::element::f32.size();
                auto out_ptr = ptr_C + C_stride;
                callBrgemm(brgemmCtx,
                           brgKernels[getBrgIdx(mIdx, k, n)],
                           local_a_ptr,
                           weight_ptr,
                           out_ptr,
                           wsp);
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
        auto ptr_a = ptr_A + (mb * M_blk * lda) * srcType.size();
        auto ptr_c = ptr_C + (mb * M_blk * ldc) * ov::element::f32.size();
        executeGemm(is_M_tail, ptr_a, scratch_b, wsp, ptr_c, scratch_a);
    }
}
void BrgemmKernel::callBrgemm(brgemmCtx& ctx,
                              std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& brgKernel,
                              const void* pin0,
                              const void* pin1,
                              void* pout,
                              void* wsp) {
    if (ctx.is_with_amx)
        amx_tile_configure(ctx.palette);
    if (ctx.is_with_comp) {
        brgemm_post_ops_data_t post_ops_data;
        brgemm_kernel_execute_postops(brgKernel.get(), 1, pin0, pin1, nullptr, pout, pout, post_ops_data, wsp);
    } else {
        brgemm_batch_element_t addr_batch;
        addr_batch.ptr.A = pin0;
        addr_batch.ptr.B = pin1;
        brgemm_kernel_execute(brgKernel.get(), 1, &addr_batch, pout, wsp, nullptr);
    }
}

}  // namespace intel_cpu
}  // namespace ov
