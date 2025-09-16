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

#include "emitters/utils.hpp"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/utils/utils.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::aarch64 {

bool GemmCopyBKernelKaiConfig::operator==(const GemmCopyBKernelKaiConfig& rhs) const {
    return m_N == rhs.m_N && m_K == rhs.m_K && m_copy_b_wei_stride == rhs.m_copy_b_wei_stride &&
           m_copy_b_col_stride == rhs.m_copy_b_col_stride && precision == rhs.precision && m_hash == rhs.m_hash;
}

bool GemmCopyBKernelKaiConfig::is_completed() const {
    return !ov::snippets::utils::any_of(0UL, m_N, m_K, m_copy_b_wei_stride) || is_empty();
}

bool GemmCopyBKernelKaiConfig::is_empty() const {
    return all_of(0UL, m_N, m_K, m_copy_b_wei_stride);
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

void GemmCopyBKernelKaiConfig::update(size_t N,
                                      size_t K,
                                      size_t row_stride_bytes,
                                      size_t col_stride_bytes,
                                      ov::element::Type prc) {
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
    precision = prc;
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

GemmCopyBKaiKernelExecutor::GemmCopyBKaiKernelExecutor(GemmCopyBKernelKaiConfig config)
    : snippets::KernelExecutor<GemmCopyBKernelKaiConfig, GemmCopyBCompiledKernel>(std::move(config)) {}

void GemmCopyBKaiKernelExecutor::update_kernel([[maybe_unused]] const GemmCopyBKernelKaiConfig& config,
                                               std::shared_ptr<GemmCopyBCompiledKernel>& kernel) const {
    if (kernel == nullptr) {
        // GemmCopyBCompiledKernel is an universal kernel, which could be used in any config and shape.
        kernel = std::make_shared<GemmCopyBCompiledKernel>();
        kernel->bias_buffer->resize(GemmCopyBKernelKaiConfig::get_N_blk() * sizeof(float), 0);
    }
}

void GemmCopyBKaiKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                               [[maybe_unused]] const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                               GemmCopyBKernelKaiConfig& config) const {
    const auto& in0_shape = snippets::utils::get_planar_vdims(expr->get_input_port(0));
    const auto N = *in0_shape.rbegin();
    const auto K = *++in0_shape.rbegin();
    const auto& prc = expr->get_node()->get_input_element_type(0);
    const auto row_stride_bytes = snippets::utils::get_dim_stride(expr->get_input_port(0), 1) * prc.size();
    const auto col_stride_bytes = snippets::utils::get_dim_stride(expr->get_input_port(0), 0) * prc.size();
    config.update(N, K, row_stride_bytes, col_stride_bytes, prc);
}

// regarding K*N(32*516),
// for K*N(32*512) part and nb(n_block-64), repack each nb block(32*64) to nb(K+1)8nb.
// for K*N(32*4) part, roundup to (32+1)*8.
void GemmCopyBKaiKernelExecutor::execute(const GemmCopyBKaiKernelExecutor* executor, void* in0, void* out0) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    // rhs is input, rhs_packed is output
    const auto& config = static_cast<const GemmCopyBKernelKaiConfig&>(executor->get_config());
    const auto& kernel = executor->get_kernel();
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    const bool is_fp16 = (config.precision == ov::element::f16);
#else
    const bool is_fp16 = false;
#endif
    fprintf(stderr, "gemm_copy_b is_fp16: %d\n", is_fp16);
    const auto K = config.get_K();                                   // K
    const auto N = config.get_N();                                   // N-rhs_stride
    const auto copy_b_wei_stride = config.get_copy_b_wei_stride();   // RHS row stride in bytes
    const auto copy_b_col_stride = config.get_copy_b_col_stride();   // RHS column stride in bytes
    const auto& n_blk_size = GemmCopyBKernelKaiConfig::get_N_blk();  // n_blk
    size_t nr = 0, kr = 0, sr = 0;
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    if (is_fp16) {
        const auto& uk = *kernel->copy_b_ukernel_f16;
        nr = uk.get_nr();
        kr = uk.get_kr();
        sr = uk.get_sr();
    } else
#endif
    {
        const auto& uk = *kernel->copy_b_ukernel_f32;
        nr = uk.get_nr();
        kr = uk.get_kr();
        sr = uk.get_sr();
    }
    size_t n_blocks = ov::snippets::utils::div_up(N, n_blk_size);
    const size_t bias_elem_size = is_fp16 ? sizeof(uint16_t) : sizeof(float);
    if (kernel->bias_buffer->size() != GemmCopyBKernelKaiConfig::get_N_blk() * bias_elem_size) {
        kernel->bias_buffer->assign(GemmCopyBKernelKaiConfig::get_N_blk() * bias_elem_size, 0);
    }
    for (size_t n_block = 0; n_block < n_blocks; n_block++) {
        size_t n_start = n_block * n_blk_size;
        size_t n_end = std::min(n_start + n_blk_size, N);
        size_t n_step = n_end - n_start;
        int8_t* src_ptr = static_cast<int8_t*>(in0) + static_cast<size_t>(n_start) * copy_b_col_stride;
        int8_t* dst_base = static_cast<int8_t*>(out0);
        if (
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
            is_fp16
#else
            false
#endif
        ) {
            const auto& uk = *kernel->copy_b_ukernel_f16;
            const size_t packed_off = uk.get_rhs_packed_offset(n_start, K);
            int8_t* dst_ptr = dst_base + packed_off;
            kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(1,
                                                              n_step,
                                                              K,
                                                              nr,
                                                              kr,
                                                              sr,                           // Packing arguments
                                                              copy_b_wei_stride,            // RHS stride in bytes
                                                              src_ptr,                      // RHS
                                                              kernel->bias_buffer->data(),  // bias
                                                              nullptr,                      // Scale
                                                              dst_ptr,                      // RHS packed
                                                              0,
                                                              nullptr);
        } else {
            const auto& uk = *kernel->copy_b_ukernel_f32;
            const size_t packed_off = uk.get_rhs_packed_offset(n_start, K);
            int8_t* dst_ptr = dst_base + packed_off;
            kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(1,
                                                             n_step,
                                                             K,
                                                             nr,
                                                             kr,
                                                             sr,                           // Packing arguments
                                                             copy_b_wei_stride,            // RHS stride in bytes
                                                             src_ptr,                      // RHS
                                                             kernel->bias_buffer->data(),  // bias
                                                             nullptr,                      // Scale
                                                             dst_ptr,                      // RHS packed
                                                             0,
                                                             nullptr);
        }
    }
}

}  // namespace ov::intel_cpu::aarch64
