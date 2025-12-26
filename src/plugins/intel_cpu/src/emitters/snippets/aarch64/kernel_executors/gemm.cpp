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
#include "openvino/core/type/float16.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu::aarch64 {

void GemmKernelKaiConfig::update(int64_t M, int64_t N, int64_t K, int64_t LDA, int64_t LDB, int64_t LDC, float beta) {
    BrgemmGenericKernelConfig::update(M, N, K, LDA, LDB, LDC, beta);
    m_hash = BrgemmGenericKernelConfig::compute_hash();
}

bool GemmKernelKaiConfig::operator==(const GemmKernelKaiConfig& rhs) const {
    return BrgemmGenericKernelConfig::operator==(rhs) && m_hash == rhs.m_hash;
}

void GemmKaiKernelExecutorBase::update_config_common(const ov::snippets::lowered::ExpressionPtr& expr,
                                                     const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                                     GemmKernelKaiConfig& config) {
    const auto [M, N, K, beta, LDC] = BrgemmKernelExecutorHelper::get_runtime_brgemm_params(expr, linear_ir);
    const auto LDA = snippets::utils::get_dim_stride(expr->get_input_port(0));
    const auto LDB = snippets::utils::get_dim_stride(expr->get_input_port(1));
    config.update(M, N, K, LDA, LDB, LDC, beta);
}

template <typename KernelT>
void GemmKaiKernelExecutorBase::ensure_kernel(std::shared_ptr<KernelT>& kernel) {
    if (kernel == nullptr) {
        // Universal kernel could be used in any config and shape, as executed piece by piece as binary call.
        kernel = std::make_shared<KernelT>();
    }
}

template <typename UkernelT>
static void execute_common_impl(const GemmKernelKaiConfig& config,
                                const GemmKaiCallArgs* args,
                                const UkernelT& ukernel,
                                size_t elem_size,
                                float clamp_min,
                                float clamp_max) {
    const auto& M = config.get_M();
    const auto& N = config.get_N();
    const auto& K = config.get_K();
    const auto& lda = config.get_LDA();
    const auto& ldc = config.get_LDC();
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
                           clamp_min,
                           clamp_max);
    }
}

GemmF32KaiKernelExecutor::GemmF32KaiKernelExecutor(GemmKernelKaiConfig config) : KernelExecutor(std::move(config)) {}

void GemmF32KaiKernelExecutor::update_kernel([[maybe_unused]] const GemmKernelKaiConfig& config,
                                             std::shared_ptr<GemmCompiledKernelF32>& kernel) const {
    ensure_kernel(kernel);
}

void GemmF32KaiKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                             const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                             GemmKernelKaiConfig& config) const {
    const auto& prc = expr->get_node()->get_input_element_type(0);
    OV_CPU_JIT_EMITTER_ASSERT(prc == ov::element::f32, "Unexpected precision for GemmF32 executor");
    update_config_common(expr, linear_ir, config);
}

void GemmF32KaiKernelExecutor::execute(const GemmF32KaiKernelExecutor* executor, const call_args* args) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    OV_CPU_JIT_EMITTER_ASSERT(args, "has nullptr args");
    execute_common_impl(static_cast<const GemmKernelKaiConfig&>(executor->get_config()),
                        args,
                        *executor->get_kernel()->gemm_ukernel,
                        sizeof(float),
                        std::numeric_limits<float>::lowest(),
                        std::numeric_limits<float>::max());
}

GemmF16KaiKernelExecutor::GemmF16KaiKernelExecutor(GemmKernelKaiConfig config) : KernelExecutor(std::move(config)) {}

void GemmF16KaiKernelExecutor::update_kernel([[maybe_unused]] const GemmKernelKaiConfig& config,
                                             std::shared_ptr<GemmCompiledKernelF16>& kernel) const {
    ensure_kernel(kernel);
}

void GemmF16KaiKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                             const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                             GemmKernelKaiConfig& config) const {
    const auto& prc = expr->get_node()->get_input_element_type(0);
    OV_CPU_JIT_EMITTER_ASSERT(prc == ov::element::f16, "Unexpected precision for GemmF16 executor");
    update_config_common(expr, linear_ir, config);
}

void GemmF16KaiKernelExecutor::execute(const GemmF16KaiKernelExecutor* executor, const call_args* args) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    OV_CPU_JIT_EMITTER_ASSERT(args, "has nullptr args");
    // Despite using an FP16 micro-kernel, the clamp bounds are kept at FP32 min/max on purpose.
    // Clamping to the FP16 dynamic range here would introduce additional saturation on top of the
    // final FP16 conversion and may lead to avoidable accuracy loss.
    execute_common_impl(static_cast<const GemmKernelKaiConfig&>(executor->get_config()),
                        args,
                        *executor->get_kernel()->gemm_ukernel,
                        sizeof(ov::float16),
                        std::numeric_limits<float>::lowest(),
                        std::numeric_limits<float>::max());
}

}  // namespace ov::intel_cpu::aarch64
