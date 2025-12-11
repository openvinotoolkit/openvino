// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
#    define __ARM_FEATURE_FP16_SCALAR_ARITHMETIC  // NOLINT(bugprone-reserved-identifier)
#endif
#ifndef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#    define __ARM_FEATURE_FP16_VECTOR_ARITHMETIC  // NOLINT(bugprone-reserved-identifier)
#endif

#include <memory>

#include "emitters/snippets/brgemm_generic.hpp"
#include "emitters/utils.hpp"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p_interface.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"

namespace ov::intel_cpu::aarch64 {

struct GemmKernelKaiConfig : public BrgemmGenericKernelConfig {
public:
    GemmKernelKaiConfig() = default;

    bool operator==(const GemmKernelKaiConfig& rhs) const;
    bool operator!=(const GemmKernelKaiConfig& rhs) const {
        return !(*this == rhs);
    }

    void update(int64_t M, int64_t N, int64_t K, int64_t LDA, int64_t LDB, int64_t LDC, float beta) override;

    [[nodiscard]] std::unique_ptr<snippets::KernelExecutorBase::GenericConfig> get_clone_ptr() const override {
        return std::make_unique<GemmKernelKaiConfig>(*this);
    }

    [[nodiscard]] size_t hash() const override {
        return m_hash;
    }

private:
    size_t m_hash{SIZE_MAX};
};

struct GemmCompiledKernelF32 {
    std::shared_ptr<kai_matmul_clamp_f32_f32_f32p_ukernel> gemm_ukernel =
        std::make_shared<kai_matmul_clamp_f32_f32_f32p_ukernel>(ukernel);

    static constexpr kai_matmul_clamp_f32_f32_f32p_ukernel ukernel{
        kai_get_m_step_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
        kai_get_n_step_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
        kai_get_nr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
        kai_get_kr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
        kai_get_sr_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
        kai_get_lhs_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
        kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
        kai_get_dst_offset_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
        kai_get_dst_size_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla,
        kai_run_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla};
};

struct GemmCompiledKernelF16 {
    std::shared_ptr<kai_matmul_clamp_f16_f16_f16p_ukernel> gemm_ukernel =
        std::make_shared<kai_matmul_clamp_f16_f16_f16p_ukernel>(ukernel);

    static constexpr kai_matmul_clamp_f16_f16_f16p_ukernel ukernel{
        kai_get_m_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_n_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_nr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_kr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_sr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_lhs_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_dst_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_get_dst_size_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
        kai_run_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla};
};

struct GemmKaiCallArgs {
    const void* A;
    const void* B;
    void* C;
};

class GemmKaiKernelExecutorBase {
protected:
    GemmKaiKernelExecutorBase() = default;
    ~GemmKaiKernelExecutorBase() = default;

    static void update_config_common(const ov::snippets::lowered::ExpressionPtr& expr,
                                     const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                     GemmKernelKaiConfig& config);

    template <typename KernelT>
    static void ensure_kernel(std::shared_ptr<KernelT>& kernel);
};

class GemmF32KaiKernelExecutor : public GemmKaiKernelExecutorBase,
                                 public snippets::KernelExecutor<GemmKernelKaiConfig, GemmCompiledKernelF32> {
public:
    using call_args = GemmKaiCallArgs;
    GemmF32KaiKernelExecutor(GemmKernelKaiConfig config);
    void update_kernel(const GemmKernelKaiConfig& config,
                       std::shared_ptr<GemmCompiledKernelF32>& kernel) const override final;
    static void execute(const GemmF32KaiKernelExecutor* executor, const call_args* args);

private:
    void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                       const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                       GemmKernelKaiConfig& config) const override;
};

class GemmF16KaiKernelExecutor : public GemmKaiKernelExecutorBase,
                                 public snippets::KernelExecutor<GemmKernelKaiConfig, GemmCompiledKernelF16> {
public:
    using call_args = GemmKaiCallArgs;
    GemmF16KaiKernelExecutor(GemmKernelKaiConfig config);
    void update_kernel(const GemmKernelKaiConfig& config,
                       std::shared_ptr<GemmCompiledKernelF16>& kernel) const override final;
    static void execute(const GemmF16KaiKernelExecutor* executor, const call_args* args);

private:
    void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                       const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                       GemmKernelKaiConfig& config) const override;
};

}  // namespace ov::intel_cpu::aarch64
