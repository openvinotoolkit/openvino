// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_copy_b.hpp"

#include <algorithm>
#include <common/utils.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "emitters/utils.hpp"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"
#include "openvino/core/type/element_type.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu::aarch64 {

bool GemmCopyBKernelKaiConfig::operator==(const GemmCopyBKernelKaiConfig& rhs) const {
    return m_N == rhs.m_N && m_K == rhs.m_K && m_copy_b_wei_stride == rhs.m_copy_b_wei_stride &&
           m_copy_b_col_stride == rhs.m_copy_b_col_stride && m_hash == rhs.m_hash;
}

bool GemmCopyBKernelKaiConfig::is_completed() const {
    return !ov::snippets::utils::any_of(0UL, m_N, m_K, m_copy_b_wei_stride) || is_empty();
}

bool GemmCopyBKernelKaiConfig::is_empty() const {
    return ov::snippets::utils::all_of(0UL, m_N, m_K, m_copy_b_wei_stride);
}

#ifdef SNIPPETS_DEBUG_CAPS
#    define PRINT(X) ss << #X << " = " << (X) << "\n"
std::string GemmCopyBKernelKaiConfig::to_string() const {
    std::stringstream ss;
    PRINT(m_N);
    PRINT(m_K);
    PRINT(m_copy_b_wei_stride);
    return ss.str();
}
#    undef PRINT
#endif

void GemmCopyBKernelKaiConfig::update(size_t N, size_t K, size_t row_stride_bytes, size_t col_stride_bytes) {
    // If one of the dims is zero, it means that GemmCopyB won't be executed (in Loop with work_amount = 0, for
    // example) To process this case, we have to make this Config as empty (nullify runtime parameters)
    if (ov::snippets::utils::any_of(0UL, N, K)) {
        m_N = 0;
        m_K = 0;
        m_copy_b_wei_stride = 0;
        m_copy_b_col_stride = 0;
    } else {
        m_N = N;
        m_K = K;
        m_copy_b_wei_stride = row_stride_bytes;
        m_copy_b_col_stride = col_stride_bytes;
    }
    m_hash = compute_hash();
}

size_t GemmCopyBKernelKaiConfig::compute_hash() const {
    size_t seed = 0;
    seed = dnnl::impl::hash_combine(seed, m_N);
    seed = dnnl::impl::hash_combine(seed, m_K);
    seed = dnnl::impl::hash_combine(seed, m_copy_b_wei_stride);
    seed = dnnl::impl::hash_combine(seed, m_copy_b_col_stride);
    return seed;
}

void GemmCopyBKaiKernelExecutorBase::update_config_common(
    const ov::snippets::lowered::ExpressionPtr& expr,
    [[maybe_unused]] const ov::snippets::lowered::LinearIRCPtr& linear_ir,
    GemmCopyBKernelKaiConfig& config) {
    const auto& in0_shape = snippets::utils::get_planar_vdims(expr->get_input_port(0));
    const auto N = *in0_shape.rbegin();
    const auto K = *++in0_shape.rbegin();
    const auto& prc = expr->get_node()->get_input_element_type(0);
    const auto row_stride_bytes = snippets::utils::get_dim_stride(expr->get_input_port(0), 1) * prc.size();
    const auto col_stride_bytes = snippets::utils::get_dim_stride(expr->get_input_port(0), 0) * prc.size();
    config.update(N, K, row_stride_bytes, col_stride_bytes);
}

template <typename CompiledKernelT>
void GemmCopyBKaiKernelExecutorBase::ensure_kernel(std::shared_ptr<CompiledKernelT>& kernel, size_t bias_elem_size) {
    const auto expected_bias_size = GemmCopyBKernelKaiConfig::get_N_blk() * bias_elem_size;
    if (kernel == nullptr) {
        kernel = std::make_shared<CompiledKernelT>();
        kernel->bias_buffer->assign(expected_bias_size, 0);
    } else if (kernel->bias_buffer->size() != expected_bias_size) {
        kernel->bias_buffer->assign(expected_bias_size, 0);
    }
}

template <auto rhs_pack_kxn, typename UkernelT>
static void execute_copy_b_common(const GemmCopyBKernelKaiConfig& config,
                                  const UkernelT& uk,
                                  std::vector<uint8_t>& bias_buffer,
                                  void* in0,
                                  void* out0,
                                  size_t elem_size) {
    const auto K = config.get_K();
    const auto N = config.get_N();
    const auto copy_b_wei_stride = config.get_copy_b_wei_stride();
    const auto copy_b_col_stride = config.get_copy_b_col_stride();
    const auto& n_blk_size = GemmCopyBKernelKaiConfig::get_N_blk();
    const size_t nr = uk.get_nr();
    const size_t kr = uk.get_kr();
    const size_t sr = uk.get_sr();
    size_t n_blocks = ov::snippets::utils::div_up(N, n_blk_size);
    if (bias_buffer.size() != GemmCopyBKernelKaiConfig::get_N_blk() * elem_size) {
        bias_buffer.assign(GemmCopyBKernelKaiConfig::get_N_blk() * elem_size, 0);
    }
    for (size_t n_block = 0; n_block < n_blocks; n_block++) {
        size_t n_start = n_block * n_blk_size;
        size_t n_end = std::min(n_start + n_blk_size, N);
        size_t n_step = n_end - n_start;
        auto* src_ptr = static_cast<int8_t*>(in0) + n_start * copy_b_col_stride;
        auto* dst_base = static_cast<int8_t*>(out0);
        const size_t packed_off = uk.get_rhs_packed_offset(n_start, K);
        auto* dst_ptr = dst_base + packed_off;
        rhs_pack_kxn(1,
                     n_step,
                     K,
                     nr,
                     kr,
                     sr,
                     copy_b_wei_stride,
                     src_ptr,
                     bias_buffer.data(),
                     nullptr,
                     dst_ptr,
                     0,
                     nullptr);
    }
}

GemmCopyBF32KaiKernelExecutor::GemmCopyBF32KaiKernelExecutor(GemmCopyBKernelKaiConfig config)
    : KernelExecutor(std::move(config)) {}

void GemmCopyBF32KaiKernelExecutor::update_kernel([[maybe_unused]] const GemmCopyBKernelKaiConfig& config,
                                                  std::shared_ptr<GemmCopyBCompiledKernelF32>& kernel) const {
    ensure_kernel(kernel, sizeof(float));
}

void GemmCopyBF32KaiKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                                  const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                                  GemmCopyBKernelKaiConfig& config) const {
    const auto& prc = expr->get_node()->get_input_element_type(0);
    OV_CPU_JIT_EMITTER_ASSERT(prc == ov::element::f32, "Unexpected precision for GemmCopyB f32 executor");
    update_config_common(expr, linear_ir, config);
}

void GemmCopyBF32KaiKernelExecutor::execute(const GemmCopyBF32KaiKernelExecutor* executor, void* in0, void* out0) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    const auto& config = static_cast<const GemmCopyBKernelKaiConfig&>(executor->get_config());
    const auto& kernel = executor->get_kernel();
    execute_copy_b_common<kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon>(config,
                                                                            *kernel->copy_b_ukernel,
                                                                            *kernel->bias_buffer,
                                                                            in0,
                                                                            out0,
                                                                            sizeof(float));
}

GemmCopyBF16KaiKernelExecutor::GemmCopyBF16KaiKernelExecutor(GemmCopyBKernelKaiConfig config)
    : KernelExecutor(std::move(config)) {}

void GemmCopyBF16KaiKernelExecutor::update_kernel([[maybe_unused]] const GemmCopyBKernelKaiConfig& config,
                                                  std::shared_ptr<GemmCopyBCompiledKernelF16>& kernel) const {
    ensure_kernel(kernel, sizeof(uint16_t));
}

void GemmCopyBF16KaiKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                                  const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                                  GemmCopyBKernelKaiConfig& config) const {
    const auto& prc = expr->get_node()->get_input_element_type(0);
    OV_CPU_JIT_EMITTER_ASSERT(prc == ov::element::f16, "Unexpected precision for GemmCopyB f16 executor");
    update_config_common(expr, linear_ir, config);
}

void GemmCopyBF16KaiKernelExecutor::execute(const GemmCopyBF16KaiKernelExecutor* executor, void* in0, void* out0) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    const auto& config = static_cast<const GemmCopyBKernelKaiConfig&>(executor->get_config());
    const auto& kernel = executor->get_kernel();
    execute_copy_b_common<kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon>(config,
                                                                             *kernel->copy_b_ukernel,
                                                                             *kernel->bias_buffer,
                                                                             in0,
                                                                             out0,
                                                                             sizeof(uint16_t));
}

}  // namespace ov::intel_cpu::aarch64
