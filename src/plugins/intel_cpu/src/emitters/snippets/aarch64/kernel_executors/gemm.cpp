// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm.hpp"

#include "openvino/core/parallel.hpp"
#include "transformations/tpp/common/op/brgemm.hpp"

#define FLOAT_MAX 3.4028235e38f
#define FLOAT_MIN (-3.4028235e38f)

namespace ov::intel_cpu::aarch64 {

GemmKaiKernelExecutor::GemmKaiKernelExecutor(GemmKernelKaiConfig config)
    : snippets::KernelExecutor<GemmKernelKaiConfig, kai_matmul_clamp_f32_f32_f32p_ukernel>(std::move(config)) {}

void GemmKaiKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                          const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                          GemmKernelKaiConfig& config) const {
    // update M/N/K/beta
    int64_t M, N, K, beta;
    std::tie(M, N, K, beta) = BrgemmKernelExecutorHelper::get_runtime_brgemm_params(expr, linear_ir);

    const auto LDA = snippets::utils::get_dim_stride(expr->get_input_port(0));
    const auto LDC = snippets::utils::get_dim_stride(expr->get_output_port(0));
    const auto LDB = snippets::utils::get_dim_stride(expr->get_input_port(1));
    config.update(M, N, K, LDA, LDB, LDC, beta);
}

void GemmKaiKernelExecutor::execute(const GemmKaiKernelExecutor* executor, void* in0, void* in1, void* out0) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    // matmul for input1 and slices of repacked input2
    const auto& config = static_cast<const GemmKernelKaiConfig&>(executor->get_config());
    const auto& M = config.get_M();
    const auto& N = config.get_N();
    const auto& K = config.get_K();
    const auto& lda = config.get_LDA();
    const auto& ldc = config.get_LDC();
    const size_t BLOCK_SIZE = 8;
    size_t n_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const size_t lhs_stride = lda * sizeof(float);  // K not split, it's also K * sizeof(float)
    const size_t dst_stride_row = ldc * sizeof(float);
    const size_t dst_stride_col = sizeof(float);
    for (size_t n_block = 0; n_block < n_blocks; n_block++) {
        size_t n_start = n_block * BLOCK_SIZE;
        size_t n_end = std::min(n_start + BLOCK_SIZE, static_cast<size_t>(N));
        size_t n_block_size = n_end - n_start;
        // rhs_packed_offset is n_start*(k+1), as packed mem as 8*(K+1) blocks. If k blocked, then lda.
        const size_t rhs_packed_offset = ukernel.get_rhs_packed_offset(n_start, K);
        // m_idx is 0 as dst already point current block.
        const size_t dst_offset = ukernel.get_dst_offset(0, n_start, dst_stride_row);
        // in0, in1, out is point to current block memory, based on block loop info, and shift done in loop begin and
        // end emitters(adjusted copyb loop info as repack outside block loops).
        float* rhs_ptr = static_cast<float*>(in1) + rhs_packed_offset / sizeof(float);
        float* dst_ptr = (static_cast<float*>(out0) + dst_offset / (sizeof(float)));
        ukernel.run_matmul(M,
                           n_block_size,
                           K,
                           in0,
                           lhs_stride,
                           rhs_ptr,
                           dst_ptr,
                           dst_stride_row,
                           dst_stride_col,
                           FLOAT_MIN,
                           FLOAT_MAX);
    }
}

}  // namespace ov::intel_cpu::aarch64
