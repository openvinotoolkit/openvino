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
#include "openvino/core/except.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/utils/utils.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::aarch64 {

GemmCopyBKernelKaiConfig::GemmCopyBKernelKaiConfig(const size_t n_blk_size)
    : m_static_params(std::make_shared<StaticParams>(n_blk_size)) {
    OPENVINO_ASSERT(n_blk_size != 0, "n_blk_size can not be zero in GemmCopyBKernelKaiConfig.");
    m_hash = compute_hash();
}

bool GemmCopyBKernelKaiConfig::operator==(const GemmCopyBKernelKaiConfig& rhs) const {
    return m_N == rhs.m_N && m_K == rhs.m_K && m_hash == rhs.m_hash &&
           (m_static_params == rhs.m_static_params || *m_static_params == *(rhs.m_static_params));
}

bool GemmCopyBKernelKaiConfig::is_completed() const {
    return !ov::snippets::utils::any_of(0UL, m_N, m_K) || is_empty();
}

bool GemmCopyBKernelKaiConfig::is_empty() const {
    return all_of(0UL, m_N, m_K);
}

#ifdef SNIPPETS_DEBUG_CAPS
#    define PRINT(X) ss << #X << " = " << (X) << "\n"
std::string GemmCopyBKernelKaiConfig::to_string() const {
    std::stringstream ss;
    ss << m_static_params->to_string() << "\n";
    PRINT(m_N);
    PRINT(m_K);
    return ss.str();
}
#    undef PRINT
#endif

void GemmCopyBKernelKaiConfig::update(size_t N, size_t K) {
    // If one of the dims is zero, it means that GemmCopyB won't be executed (in Loop with work_amount = 0, for
    // example) To process this case, we have to make this Config as empty (nullify runtime parameters)
    if (ov::snippets::utils::any_of(0UL, N, K)) {
        m_N = 0;
        m_K = 0;
    } else {
        m_N = N;
        m_K = K;
    }
    m_hash = compute_hash();
}

size_t GemmCopyBKernelKaiConfig::compute_hash() const {
    size_t seed = m_static_params->hash;
    seed = dnnl::impl::hash_combine(seed, m_N);
    seed = dnnl::impl::hash_combine(seed, m_K);
    return seed;
}

GemmCopyBKernelKaiConfig::StaticParams::StaticParams(size_t wei_n_blk)
    : wei_N_blk(wei_n_blk),
      hash(init_hash(wei_N_blk)) {}

bool GemmCopyBKernelKaiConfig::StaticParams::operator==(const StaticParams& rhs) const {
    return wei_N_blk == rhs.wei_N_blk && hash == rhs.hash;
}

size_t GemmCopyBKernelKaiConfig::StaticParams::init_hash(size_t wei_n_blk) {
    size_t seed = 0;
    seed = dnnl::impl::hash_combine(seed, wei_n_blk);
    return seed;
}

#ifdef SNIPPETS_DEBUG_CAPS
#    define PRINT(X) ss << #X << " = " << (X) << "\n"
std::string GemmCopyBKernelKaiConfig::StaticParams::to_string() const {
    std::stringstream ss;
    PRINT(wei_N_blk);
    return ss.str();
}
#    undef PRINT
#endif

GemmCopyBKaiKernelExecutor::GemmCopyBKaiKernelExecutor(GemmCopyBKernelKaiConfig config)
    : snippets::KernelExecutor<GemmCopyBKernelKaiConfig, GemmCopyBCompiledKernel>(std::move(config)) {}

void GemmCopyBKaiKernelExecutor::update_kernel(const GemmCopyBKernelKaiConfig& config,
                                               std::shared_ptr<GemmCopyBCompiledKernel>& kernel) const {
    if (kernel == nullptr) {
        // GemmCopyBCompiledKernel is an universal kernel, which could be used in any config and shape.
        // 1. It's executed block by block with binary call and config passed as parameters.
        // 2. In each block, at most n_blk_size bias is needed. n_blk_size is a fixed value in gemm blocking pass.
        // if N block size changed in gemm blocking in future and based on shape, bias_buffer should be updated
        kernel = std::make_shared<GemmCopyBCompiledKernel>();
        const auto& n_blk_size = config.get_n_blk_size();
        kernel->bias_buffer->resize(n_blk_size * sizeof(float), 0);
    }
}

void GemmCopyBKaiKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                               [[maybe_unused]] const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                               GemmCopyBKernelKaiConfig& config) const {
    const auto& in0_shape = snippets::utils::get_planar_vdims(expr->get_input_port(0));
    int64_t N = *in0_shape.rbegin();
    int64_t K = *++in0_shape.rbegin();
    config.update(N, K);
}

// regarding K*N(32*516),
// for K*N(32*512) part and nb(n_block-64), repack each nb block(32*64) to nb(K+1)8nb.
// for K*N(32*4) part, roundup to (32+1)*8.
void GemmCopyBKaiKernelExecutor::execute(const GemmCopyBKaiKernelExecutor* executor, void* in0, void* out0) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    // rhs is input, rhs_packed is output
    const auto& config = static_cast<const GemmCopyBKernelKaiConfig&>(executor->get_config());
    const auto& kernel = executor->get_kernel();
    const auto& ukernel = kernel->copy_b_ukernel;
    const auto K = config.get_K();                     // K
    const auto N = config.get_N();                     // N-rhs_stride
    const auto& n_blk_size = config.get_n_blk_size();  // n_blk
    const size_t nr = ukernel->get_nr();
    const size_t kr = ukernel->get_kr();
    const size_t sr = ukernel->get_sr();
    size_t n_blocks = ov::snippets::utils::div_up(N, n_blk_size);
    size_t dst_offset = 0;
    const size_t rhsPackedSize = kai_get_rhs_packed_size_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(n_blk_size, K);
    for (size_t n_block = 0; n_block < n_blocks; n_block++) {
        size_t n_start = n_block * n_blk_size;
        size_t n_end = std::min(n_start + n_blk_size, N);
        size_t n_step = n_end - n_start;
        int8_t* src_ptr = static_cast<int8_t*>(in0) + n_start * sizeof(int32_t);
        int8_t* dst_ptr = static_cast<int8_t*>(out0) + dst_offset;
        dst_offset += rhsPackedSize;
        kai_run_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon(1,
                                                         n_step,
                                                         K,
                                                         nr,
                                                         kr,
                                                         sr,                           // Packing arguments
                                                         N * sizeof(float),            // RHS stride
                                                         src_ptr,                      // RHS
                                                         kernel->bias_buffer->data(),  // bias
                                                         nullptr,                      // Scale
                                                         dst_ptr,                      // RHS packed
                                                         0,
                                                         nullptr);
    }
}

}  // namespace ov::intel_cpu::aarch64
