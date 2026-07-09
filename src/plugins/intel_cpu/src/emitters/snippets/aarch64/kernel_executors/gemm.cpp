// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm.hpp"

#include <algorithm>
#include <common/utils.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "emitters/snippets/brgemm_generic.hpp"
#include "emitters/utils.hpp"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp_qsi8cxp_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu::aarch64 {

void GemmKernelKaiConfig::update(int64_t M, int64_t N, int64_t K, int64_t LDA, int64_t LDB, int64_t LDC, float beta) {
    BrgemmGenericKernelConfig::update(M, N, K, LDA, LDB, LDC, beta);
    m_hash = compute_hash();
}

void GemmKernelKaiConfig::set_input_a_zero_point(int32_t input_a_zero_point) {
    m_input_a_zero_point = input_a_zero_point;
    m_hash = compute_hash();
}

bool GemmKernelKaiConfig::operator==(const GemmKernelKaiConfig& rhs) const {
    return BrgemmGenericKernelConfig::operator==(rhs) && m_input_a_zero_point == rhs.m_input_a_zero_point &&
           m_hash == rhs.m_hash;
}

size_t GemmKernelKaiConfig::compute_hash() const {
    return dnnl::impl::hash_combine(BrgemmGenericKernelConfig::compute_hash(), m_input_a_zero_point);
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

kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel GemmCompiledKernelI8::get_selected_ukernel() {
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

static int8_t load_lhs_value(const uint8_t* src, size_t idx, int32_t input_a_zero_point) {
    if (input_a_zero_point == 0) {
        return reinterpret_cast<const int8_t*>(src)[idx];
    }
    return static_cast<int8_t>(static_cast<int32_t>(src[idx]) - input_a_zero_point);
}

static void pack_lhs_i8(const uint8_t* src,
                        size_t M,
                        size_t K,
                        size_t lda,
                        int32_t input_a_zero_point,
                        const kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel& ukernel,
                        uint8_t* dst) {
    const size_t mr = ukernel.get_mr();
    const size_t kr = ukernel.get_kr();
    const size_t sr = ukernel.get_sr();
    OPENVINO_ASSERT(sr != 0 && kr % sr == 0, "Unexpected KAI i8 GEMM packing parameters");

    const size_t k_block_len = kr / sr;
    const size_t k_internal = ov::snippets::utils::rnd_up(K, static_cast<size_t>(32));
    const size_t num_blocks_k_internal = k_internal / k_block_len;

    for (size_t row = 0; row < M; ++row) {
        auto* row_block = dst + kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(row, K, mr, kr, sr);
        const size_t dst_x = row % mr;
        auto* payload = row_block + dst_x * k_block_len;
        const auto* src_row = src + row * lda;

        for (size_t block = 0; block < num_blocks_k_internal; ++block) {
            for (size_t k_block_idx = 0; k_block_idx < k_block_len; ++k_block_idx) {
                const size_t k_idx = block * k_block_len + k_block_idx;
                *payload++ = k_idx < K ? static_cast<uint8_t>(load_lhs_value(src_row, k_idx, input_a_zero_point)) : 0U;
            }
            payload += (mr - 1) * k_block_len;
        }

        auto* lhs_offsets = row_block + mr * k_internal;
        reinterpret_cast<int32_t*>(lhs_offsets)[dst_x] = input_a_zero_point;
        auto* lhs_scales = lhs_offsets + mr * sizeof(int32_t);
        reinterpret_cast<float*>(lhs_scales)[dst_x] = 1.0F;
    }
}

static void execute_i8_impl(const GemmKernelKaiConfig& config,
                            const GemmKaiCallArgs* args,
                            const kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel& ukernel) {
    const auto& M = config.get_M();
    const auto& N = config.get_N();
    const auto& K = config.get_K();
    const auto& lda = config.get_LDA();
    const auto& ldc = config.get_LDC();

    const size_t packed_lhs_size =
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, ukernel.get_mr(), ukernel.get_kr(), ukernel.get_sr());
    std::vector<uint8_t> packed_lhs(packed_lhs_size);
    pack_lhs_i8(static_cast<const uint8_t*>(args->A),
                M,
                K,
                lda,
                config.get_input_a_zero_point(),
                ukernel,
                packed_lhs.data());

    const size_t block_size = ukernel.get_n_step();
    const size_t n_blocks = ov::snippets::utils::div_up(static_cast<size_t>(N), block_size);
    const size_t dst_stride_row = ldc * sizeof(float);
    for (size_t n_block = 0; n_block < n_blocks; ++n_block) {
        const size_t n_start = n_block * block_size;
        const size_t n_end = std::min(n_start + block_size, static_cast<size_t>(N));
        const size_t n_block_size = n_end - n_start;
        const size_t rhs_packed_offset = ukernel.get_rhs_packed_offset(n_start, K);
        const size_t dst_offset = ukernel.get_dst_offset(0, n_start, dst_stride_row);
        const auto* rhs_ptr = static_cast<const uint8_t*>(args->B) + rhs_packed_offset;
        auto* dst_ptr = static_cast<uint8_t*>(args->C) + dst_offset;

        ukernel.run_matmul(M,
                           n_block_size,
                           K,
                           packed_lhs.data(),
                           rhs_ptr,
                           reinterpret_cast<float*>(dst_ptr),
                           dst_stride_row,
                           sizeof(float),
                           std::numeric_limits<float>::lowest(),
                           std::numeric_limits<float>::max());
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

GemmI8KaiKernelExecutor::GemmI8KaiKernelExecutor(GemmKernelKaiConfig config) : KernelExecutor(std::move(config)) {}

void GemmI8KaiKernelExecutor::update_kernel([[maybe_unused]] const GemmKernelKaiConfig& config,
                                            std::shared_ptr<GemmCompiledKernelI8>& kernel) const {
    ensure_kernel(kernel);
}

void GemmI8KaiKernelExecutor::update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                                            const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                            GemmKernelKaiConfig& config) const {
    const auto& input_prc = expr->get_node()->get_input_element_type(0);
    const auto& weights_prc = expr->get_node()->get_input_element_type(1);
    OV_CPU_JIT_EMITTER_ASSERT(
        ov::snippets::utils::any_of(input_prc, ov::element::i8, ov::element::u8) && weights_prc == ov::element::i8,
        "Unexpected precision for GemmI8 executor");
    update_config_common(expr, linear_ir, config);
    config.set_input_a_zero_point(input_prc == ov::element::u8 ? 128 : 0);
}

void GemmI8KaiKernelExecutor::execute(const GemmI8KaiKernelExecutor* executor, const call_args* args) {
    OV_CPU_JIT_EMITTER_ASSERT(executor, "has nullptr executor");
    OV_CPU_JIT_EMITTER_ASSERT(args, "has nullptr args");
    execute_i8_impl(static_cast<const GemmKernelKaiConfig&>(executor->get_config()),
                    args,
                    *executor->get_kernel()->gemm_ukernel);
}

}  // namespace ov::intel_cpu::aarch64
