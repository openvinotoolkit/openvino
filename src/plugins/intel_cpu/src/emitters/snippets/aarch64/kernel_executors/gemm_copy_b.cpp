// Copyright (C) 2018-2026 Intel Corporation
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
#include "kai/kai_common.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp_qsi8cxp_interface.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp_qsi8cx_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x16p32x1b_x16_x16_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x32p16x1b_x32_x32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_x16p32x1bx16_x16_x16_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_x32p16x1bx32_x32_x32_neon.h"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu::aarch64 {

bool GemmCopyBKernelKaiConfig::operator==(const GemmCopyBKernelKaiConfig& rhs) const {
    return m_N == rhs.m_N && m_K == rhs.m_K && m_copy_b_wei_stride == rhs.m_copy_b_wei_stride &&
           m_copy_b_col_stride == rhs.m_copy_b_col_stride && m_is_transposed == rhs.m_is_transposed &&
           m_hash == rhs.m_hash;
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
    PRINT(m_copy_b_col_stride);
    PRINT(m_is_transposed);
    return ss.str();
}
#    undef PRINT
#endif

void GemmCopyBKernelKaiConfig::update(size_t N,
                                      size_t K,
                                      size_t row_stride_bytes,
                                      size_t col_stride_bytes,
                                      bool is_transposed) {
    // If one of the dims is zero, it means that GemmCopyB won't be executed (in Loop with work_amount = 0, for
    // example) To process this case, we have to make this Config as empty (nullify runtime parameters)
    if (ov::snippets::utils::any_of(0UL, N, K)) {
        m_N = 0;
        m_K = 0;
        m_copy_b_wei_stride = 0;
        m_copy_b_col_stride = 0;
        m_is_transposed = false;
    } else {
        m_N = N;
        m_K = K;
        m_copy_b_wei_stride = row_stride_bytes;
        m_copy_b_col_stride = col_stride_bytes;
        m_is_transposed = is_transposed;
    }
    m_hash = compute_hash();
}

size_t GemmCopyBKernelKaiConfig::compute_hash() const {
    size_t seed = 0;
    seed = dnnl::impl::hash_combine(seed, m_N);
    seed = dnnl::impl::hash_combine(seed, m_K);
    seed = dnnl::impl::hash_combine(seed, m_copy_b_wei_stride);
    seed = dnnl::impl::hash_combine(seed, m_copy_b_col_stride);
    seed = dnnl::impl::hash_combine(seed, m_is_transposed);
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
    const auto& layout = expr->get_input_port(0).get_descriptor_ptr()->get_layout();
    const auto is_transposed = snippets::utils::get_input_dim_idx(layout, 0) != layout.size() - 1;
    config.update(N, K, row_stride_bytes, col_stride_bytes, is_transposed);
}

template <typename CompiledKernelT>
void GemmCopyBKaiKernelExecutorBase::ensure_kernel(std::shared_ptr<CompiledKernelT>& kernel) {
    if (kernel == nullptr) {
        kernel = std::make_shared<CompiledKernelT>();
    }
}

kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel GemmCopyBCompiledKernelI8::get_selected_ukernel() {
    if (ov::with_cpu_arm_i8mm()) {
        return {kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
                kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
                kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
                kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
                kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
                kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
                kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
                kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
                kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
                kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm,
                kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm};
    }
    OV_CPU_JIT_EMITTER_ASSERT(ov::with_cpu_arm_dotprod(), "KAI i8 GEMM requires ARM DotProd or I8MM");
    return {kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod,
            kai_run_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod};
}

template <auto rhs_pack_kxn, auto rhs_pack_nxk, typename UkernelT>
static void execute_copy_b_common(const GemmCopyBKernelKaiConfig& config, const UkernelT& uk, void* in0, void* out0) {
    const auto K = config.get_K();
    const auto N = config.get_N();
    const auto copy_b_wei_stride = config.get_copy_b_wei_stride();
    const auto copy_b_col_stride = config.get_copy_b_col_stride();
    const auto& n_blk_size = GemmCopyBKernelKaiConfig::get_N_blk();
    const size_t nr = uk.get_nr();
    const size_t kr = uk.get_kr();
    const size_t sr = uk.get_sr();
    size_t n_blocks = ov::snippets::utils::div_up(N, n_blk_size);
    for (size_t n_block = 0; n_block < n_blocks; n_block++) {
        size_t n_start = n_block * n_blk_size;
        size_t n_end = std::min(n_start + n_blk_size, N);
        size_t n_step = n_end - n_start;
        auto* src_ptr = static_cast<int8_t*>(in0) + n_start * copy_b_col_stride;
        auto* dst_base = static_cast<int8_t*>(out0);
        const size_t packed_off = uk.get_rhs_packed_offset(n_start, K);
        auto* dst_ptr = dst_base + packed_off;
        if (config.is_transposed()) {
            rhs_pack_nxk(1, n_step, K, nr, kr, sr, copy_b_col_stride, src_ptr, nullptr, nullptr, dst_ptr, 0, nullptr);
        } else {
            rhs_pack_kxn(1, n_step, K, nr, kr, sr, copy_b_wei_stride, src_ptr, nullptr, nullptr, dst_ptr, 0, nullptr);
        }
    }
}

static std::vector<int8_t> make_dense_rhs_tile(const GemmCopyBKernelKaiConfig& config,
                                               const int8_t* src,
                                               size_t n_step) {
    const auto K = config.get_K();
    const auto copy_b_wei_stride = config.get_copy_b_wei_stride();
    const auto copy_b_col_stride = config.get_copy_b_col_stride();
    std::vector<int8_t> dense(n_step * K);

    if (config.is_transposed()) {
        for (size_t n = 0; n < n_step; ++n) {
            for (size_t k = 0; k < K; ++k) {
                dense[n * K + k] =
                    *reinterpret_cast<const int8_t*>(reinterpret_cast<const uint8_t*>(src) + n * copy_b_col_stride + k);
            }
        }
    } else {
        for (size_t k = 0; k < K; ++k) {
            for (size_t n = 0; n < n_step; ++n) {
                dense[k * n_step + n] =
                    *reinterpret_cast<const int8_t*>(reinterpret_cast<const uint8_t*>(src) + k * copy_b_wei_stride + n);
            }
        }
    }
    return dense;
}

static void execute_copy_b_i8_common(const GemmCopyBKernelKaiConfig& config,
                                     const kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel& uk,
                                     void* in0,
                                     void* out0) {
    const auto K = config.get_K();
    const auto N = config.get_N();
    const auto copy_b_col_stride = config.get_copy_b_col_stride();
    const auto& n_blk_size = GemmCopyBKernelKaiConfig::get_N_blk();
    const size_t nr = uk.get_nr();
    const size_t kr = uk.get_kr();
    const size_t sr = uk.get_sr();
    const kai_rhs_pack_qsi8cx_params params{1, 1.0F};

    const size_t n_blocks = ov::snippets::utils::div_up(N, n_blk_size);
    for (size_t n_block = 0; n_block < n_blocks; ++n_block) {
        const size_t n_start = n_block * n_blk_size;
        const size_t n_end = std::min(n_start + n_blk_size, N);
        const size_t n_step = n_end - n_start;
        const auto* src_ptr = static_cast<const int8_t*>(in0) + n_start * copy_b_col_stride;
        auto* dst_ptr = static_cast<int8_t*>(out0) + uk.get_rhs_packed_offset(n_start, K);
        const std::vector<int8_t> dense = make_dense_rhs_tile(config, src_ptr, n_step);
        const std::vector<float> scales(n_step, 1.0F);

        if (config.is_transposed()) {
            kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(1,
                                                     n_step,
                                                     K,
                                                     nr,
                                                     kr,
                                                     sr,
                                                     dense.data(),
                                                     nullptr,
                                                     scales.data(),
                                                     dst_ptr,
                                                     0,
                                                     &params);
        } else {
            kai_run_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(1,
                                                     n_step,
                                                     K,
                                                     nr,
                                                     kr,
                                                     sr,
                                                     dense.data(),
                                                     nullptr,
                                                     scales.data(),
                                                     dst_ptr,
                                                     0,
                                                     &params);
        }
    }
}

GemmCopyBF32KaiKernelExecutor::GemmCopyBF32KaiKernelExecutor(GemmCopyBKernelKaiConfig config)
    : KernelExecutor(std::move(config)) {}

void GemmCopyBF32KaiKernelExecutor::update_kernel([[maybe_unused]] const GemmCopyBKernelKaiConfig& config,
                                                  std::shared_ptr<GemmCopyBCompiledKernelF32>& kernel) const {
    ensure_kernel(kernel);
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
    execute_copy_b_common<kai_run_rhs_pack_kxn_x32p16x1b_x32_x32_neon, kai_run_rhs_pack_nxk_x32p16x1bx32_x32_x32_neon>(
        config,
        *kernel->copy_b_ukernel,
        in0,
        out0);
}

GemmCopyBF16KaiKernelExecutor::GemmCopyBF16KaiKernelExecutor(GemmCopyBKernelKaiConfig config)
    : KernelExecutor(std::move(config)) {}

void GemmCopyBF16KaiKernelExecutor::update_kernel([[maybe_unused]] const GemmCopyBKernelKaiConfig& config,
                                                  std::shared_ptr<GemmCopyBCompiledKernelF16>& kernel) const {
    ensure_kernel(kernel);
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
    execute_copy_b_common<kai_run_rhs_pack_kxn_x16p32x1b_x16_x16_neon, kai_run_rhs_pack_nxk_x16p32x1bx16_x16_x16_neon>(
        config,
        *kernel->copy_b_ukernel,
        in0,
        out0);
}

GemmCopyBI8KaiKernelExecutor::GemmCopyBI8KaiKernelExecutor(GemmCopyBKernelKaiConfig config)
    : KernelExecutor(std::move(config)) {}

void GemmCopyBI8KaiKernelExecutor::update_kernel([[maybe_unused]] const GemmCopyBKernelKaiConfig& config,
                                                 std::shared_ptr<GemmCopyBCompiledKernelI8>& kernel) const {
    ensure_kernel(kernel);
}

void GemmCopyBI8KaiKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                                 const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                                 GemmCopyBKernelKaiConfig& config) const {
    const auto& prc = expr->get_node()->get_input_element_type(0);
    OV_CPU_JIT_EMITTER_ASSERT(prc == ov::element::i8, "Unexpected precision for GemmCopyB i8 executor");
    update_config_common(expr, linear_ir, config);
}

void GemmCopyBI8KaiKernelExecutor::execute(const GemmCopyBI8KaiKernelExecutor* executor, void* in0, void* out0) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    const auto& config = static_cast<const GemmCopyBKernelKaiConfig&>(executor->get_config());
    const auto& kernel = executor->get_kernel();
    execute_copy_b_i8_common(config, *kernel->copy_b_ukernel, in0, out0);
}

}  // namespace ov::intel_cpu::aarch64
