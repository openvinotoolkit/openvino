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

#include <common/utils.hpp>

#include "emitters/snippets/brgemm_generic.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu::aarch64 {

void GemmKernelKaiConfig::update(int64_t M,
                                 int64_t N,
                                 int64_t K,
                                 int64_t LDA,
                                 int64_t LDB,
                                 int64_t LDC,
                                 float beta,
                                 ov::element::Type prc) {
    BrgemmGenericKernelConfig::update(M, N, K, LDA, LDB, LDC, beta);
    precision = prc;
    m_hash = BrgemmGenericKernelConfig::compute_hash();
    m_hash = dnnl::impl::hash_combine(m_hash, precision.hash());
}

bool GemmKernelKaiConfig::operator==(const GemmKernelKaiConfig& rhs) const {
    return BrgemmGenericKernelConfig::operator==(rhs) && precision == rhs.precision && m_hash == rhs.m_hash;
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
    const auto& prc = expr->get_node()->get_input_element_type(0);
    config.update(M, N, K, LDA, LDB, LDC, beta, prc);
}

void GemmKaiKernelExecutor::execute(const GemmKaiKernelExecutor* executor, const call_args* args) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    OV_CPU_JIT_EMITTER_ASSERT(args, "has nullptr args");

    const auto& config = static_cast<const GemmKernelKaiConfig&>(executor->get_config());
    const auto& kernel = executor->get_kernel();
    const auto& M = config.get_M();
    const auto& N = config.get_N();
    const auto& K = config.get_K();
    const auto& lda = config.get_LDA();
    const auto& ldc = config.get_LDC();
    const bool is_fp16 = config.precision == ov::element::f16;
    if (is_fp16) {
        const auto& ukernel = *kernel->gemm_ukernel_f16;
        const size_t elem_size = sizeof(uint16_t);
        const size_t BLOCK_SIZE = ukernel.get_n_step();
        size_t n_blocks = ov::snippets::utils::div_up(static_cast<size_t>(N), BLOCK_SIZE);
        const size_t lhs_stride = lda * elem_size;
        const size_t dst_stride_row = ldc * elem_size;
        const size_t dst_stride_col = elem_size;
        for (size_t n_block = 0; n_block < n_blocks; n_block++) {
            size_t n_start = n_block * BLOCK_SIZE;
            size_t n_end = std::min(n_start + BLOCK_SIZE, static_cast<size_t>(N));
            size_t n_block_size = n_end - n_start;
            const size_t rhs_packed_offset = ukernel.get_rhs_packed_offset(n_start, K);
            const size_t dst_offset = ukernel.get_dst_offset(0, n_start, dst_stride_row);
            const uint8_t* rhs_ptr = static_cast<const uint8_t*>(args->B) + rhs_packed_offset;
            uint8_t* dst_ptr = static_cast<uint8_t*>(args->C) + dst_offset;
            ukernel.run_matmul(M,
                               n_block_size,
                               K,
                               args->A,
                               lhs_stride,
                               rhs_ptr,
                               dst_ptr,
                               dst_stride_row,
                               dst_stride_col,
                               std::numeric_limits<float>::lowest(),
                               std::numeric_limits<float>::max());
        }
        return;
    }
    {
        const auto& ukernel = *kernel->gemm_ukernel_f32;
        const size_t elem_size = sizeof(float);
        const size_t BLOCK_SIZE = ukernel.get_n_step();
        size_t n_blocks = ov::snippets::utils::div_up(static_cast<size_t>(N), BLOCK_SIZE);
        const size_t lhs_stride = lda * elem_size;
        const size_t dst_stride_row = ldc * elem_size;
        const size_t dst_stride_col = elem_size;
        for (size_t n_block = 0; n_block < n_blocks; n_block++) {
            size_t n_start = n_block * BLOCK_SIZE;
            size_t n_end = std::min(n_start + BLOCK_SIZE, static_cast<size_t>(N));
            size_t n_block_size = n_end - n_start;
            const size_t rhs_packed_offset = ukernel.get_rhs_packed_offset(n_start, K);
            const size_t dst_offset = ukernel.get_dst_offset(0, n_start, dst_stride_row);
            const uint8_t* rhs_ptr = static_cast<const uint8_t*>(args->B) + rhs_packed_offset;
            uint8_t* dst_ptr = static_cast<uint8_t*>(args->C) + dst_offset;
            ukernel.run_matmul(M,
                               n_block_size,
                               K,
                               args->A,
                               lhs_stride,
                               rhs_ptr,
                               dst_ptr,
                               dst_stride_row,
                               dst_stride_col,
                               std::numeric_limits<float>::lowest(),
                               std::numeric_limits<float>::max());
        }
    }
}

}  // namespace ov::intel_cpu::aarch64
