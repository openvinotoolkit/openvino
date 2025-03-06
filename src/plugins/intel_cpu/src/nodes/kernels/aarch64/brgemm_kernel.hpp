// Copyright (C) 2018-2025 Intel Corporation
// Copyright (C) 2024 FUJITSU LIMITED
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cpu/aarch64/brgemm/brgemm.hpp>
#include <cpu/aarch64/matmul/brgemm_matmul_copy_utils.hpp>
#include <cpu/aarch64/matmul/brgemm_matmul_utils.hpp>
#include <cstddef>
#include <openvino/core/type/element_type.hpp>

namespace ov::intel_cpu {

class BrgemmKernel {
public:
    // Construct brgemm kernel for matmul (M, K) * (K, N)/(N, K)^T
    // FP32 * FP32 -> FP32
    // lda is the leading dimension for A matrix
    // ldb is the leading dimension for B matrix
    // ldc is the leading dimension for C matrix
    // b_transpose indicates wheter B matrix is transposed.
    BrgemmKernel(size_t M,
                 size_t N,
                 size_t K,
                 size_t lda,
                 size_t ldb,
                 size_t ldc,
                 bool b_transposed = false,
                 ov::element::Type inType = ov::element::f32,
                 bool b_accumulate = false);
    // execute all M
    void executeGemm(void* a, void* b, void* c, void* wsp, void* scratch_a, void* scratch_b);
    // execute by m_blk
    void executeGemm(bool is_M_tail, void* a, void* b, void* c, void* wsp, void* scratch_a);

    void copy_buffer_b(void* b, void* scratch_b);
    // bytes needed to place scratch buffer a
    [[nodiscard]] const size_t get_scratch_a_size() const;
    // bytes needed to place scratch buffer b
    [[nodiscard]] const size_t get_scratch_b_size() const;
    [[nodiscard]] const size_t get_wsp_size() const {
        return 4 * 1024;
    }

private:
    size_t M = 0, M_blk = 0, M_tail = 0;
    size_t K = 0, K_blk = 0, K_tail = 0, N = 0, N_blk = 0, N_tail = 0;
    size_t lda = 0, ldb = 0, ldc = 0;
    bool b_transposed = false;
    size_t kBlkStep = 0;
    size_t packedBSize = 0;
    size_t packedASize = 0;
    ov::element::Type inType;
    static constexpr size_t MHA_BRGEMM_KERNELS_NUM = 8;
    static constexpr size_t matmulOptimalM = 32;
    struct brgemmCtx {
        size_t M = 0, N = 0, K = 0, LDA = 0, LDB = 0, LDC = 0;
        dnnl_data_type_t dt_in0 = dnnl_data_type_undef;
        dnnl_data_type_t dt_in1 = dnnl_data_type_undef;
        bool transpose_a = false;
        bool transpose_b = false;
        float beta = 0.0f;
    };
    brgemmCtx brgCtxs[MHA_BRGEMM_KERNELS_NUM];
    std::unique_ptr<dnnl::impl::cpu::aarch64::brgemm_kernel_t> brgKernels[MHA_BRGEMM_KERNELS_NUM];
    std::unique_ptr<dnnl::impl::cpu::aarch64::matmul::jit_brgemm_matmul_copy_a_t> brgCopyAKernel;
    std::unique_ptr<dnnl::impl::cpu::aarch64::matmul::jit_brgemm_matmul_copy_b_t> brgCopyBKernel;
    size_t getBrgIdx(size_t mIdx, size_t kIdx, size_t nIdx) {
        return mIdx * 4 + kIdx * 2 + nIdx;
    }
    void init_brgemm(brgemmCtx& ctx, std::unique_ptr<dnnl::impl::cpu::aarch64::brgemm_kernel_t>& brgKernel);
    // LDA, LDB is used for stride of target memory
    void init_brgemm_copy_a(
        std::unique_ptr<dnnl::impl::cpu::aarch64::matmul::jit_brgemm_matmul_copy_a_t>& brgCopyKernel,
        size_t K,
        size_t K_blk,
        size_t K_tail,
        size_t LDA,
        dnnl_data_type_t dt_in0,
        bool transpose = false,
        size_t copy_A_src_stride = 0);

    void init_brgemm_copy_b(
        std::unique_ptr<dnnl::impl::cpu::aarch64::matmul::jit_brgemm_matmul_copy_b_t>& brgCopyKernel,
        size_t N,
        size_t N_blk,
        size_t N_tail,
        size_t LDB,
        size_t K,
        dnnl_data_type_t dt_in0,
        dnnl_data_type_t dt_in1,
        bool transpose = false,
        size_t copy_B_wei_stride = 0);

    void callBrgemm(brgemmCtx& ctx,
                    std::unique_ptr<dnnl::impl::cpu::aarch64::brgemm_kernel_t>& brgKernel,
                    const void* pin0,
                    const void* pin1,
                    void* pout,
                    void* wsp);
};
}  // namespace ov::intel_cpu
