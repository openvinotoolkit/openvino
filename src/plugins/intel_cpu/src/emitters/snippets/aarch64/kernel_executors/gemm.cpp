// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

#include "emitters/snippets/brgemm_generic.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/aarch64/op/gemm_utils.hpp"

namespace ov::intel_cpu::aarch64 {

void GemmKernelKaiConfig::update(int64_t M, int64_t N, int64_t K, int64_t LDA, int64_t LDB, int64_t LDC, float beta) {
    BrgemmGenericKernelConfig::update(M, N, K, LDA, LDB, LDC, beta);
    m_hash = BrgemmGenericKernelConfig::compute_hash();
}

bool GemmKernelKaiConfig::operator==(const GemmKernelKaiConfig& rhs) const {
    return BrgemmGenericKernelConfig::operator==(rhs) && m_hash == rhs.m_hash;
}

GemmKaiKernelExecutor::GemmKaiKernelExecutor(GemmKernelKaiConfig config)
    : snippets::KernelExecutor<GemmKernelKaiConfig, GemmCompiledKernel>(std::move(config)) {}

void GemmKaiKernelExecutor::update_kernel([[maybe_unused]] const GemmKernelKaiConfig& config,
                                          std::shared_ptr<GemmCompiledKernel>& kernel) const {
    if (kernel == nullptr) {
        // universal kernel could be used in any config and shape, as excuted peice by peice as binary call.
        // config is passed as binary call parameters.
        kernel = std::make_shared<GemmCompiledKernel>();
    }
}

void GemmKaiKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                          const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                          GemmKernelKaiConfig& config) const {
    const auto [M, N, K, beta] = BrgemmKernelExecutorHelper::get_runtime_brgemm_params(expr, linear_ir);

    const auto LDA = snippets::utils::get_dim_stride(expr->get_input_port(0));
    const auto LDC = snippets::utils::get_dim_stride(expr->get_output_port(0));
    const auto LDB = snippets::utils::get_dim_stride(expr->get_input_port(1));
    config.update(M, N, K, LDA, LDB, LDC, beta);
}

void GemmKaiKernelExecutor::execute(const GemmKaiKernelExecutor* executor, void* in0, void* in1, void* out0) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    // matmul for input1 and slices of repacked input2
    const auto& config = static_cast<const GemmKernelKaiConfig&>(executor->get_config());
    const auto& kernel = executor->get_kernel();
    const auto& ukernel = *kernel->gemm_ukernel;
    const auto& M = config.get_M();
    const auto& N = config.get_N();
    const auto& K = config.get_K();
    const auto& lda = config.get_LDA();
    const auto& ldc = config.get_LDC();
    const size_t& BLOCK_SIZE = ov::intel_cpu::aarch64::gemm_utils::repacking::get_inner_n_block(element::f32);
    size_t n_blocks = ov::snippets::utils::div_up(N, BLOCK_SIZE);
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
                           std::numeric_limits<float>::min(),
                           std::numeric_limits<float>::max());
    }
}

void GemmKaiKernelExecutor::execute(const GemmKaiKernelExecutor* executor, const call_args* args) {
    if (!executor || !args) {
        return;
    }

    execute(executor, const_cast<void*>(args->A), const_cast<void*>(args->B), args->C);
}

}  // namespace ov::intel_cpu::aarch64
