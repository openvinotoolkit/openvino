// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/snippets/brgemm_generic.hpp"
#include "emitters/utils.hpp"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p_interface.h"
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

struct GemmCompiledKernel {
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

class GemmKaiKernelExecutor : public snippets::KernelExecutor<GemmKernelKaiConfig, GemmCompiledKernel> {
public:
    struct call_args {
        const void* A;
        const void* B;
        void* C;
    };

    GemmKaiKernelExecutor(GemmKernelKaiConfig config);

    // No need kernel update, just update config is enough for update. The universal ukernel is reused with any config.
    void update_kernel(const GemmKernelKaiConfig& config,
                       std::shared_ptr<GemmCompiledKernel>& kernel) const override final;
    // ABI-compliant execute function that takes call_args structure
    static void execute(const GemmKaiKernelExecutor* executor, const call_args* args);

private:
    void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                       const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                       GemmKernelKaiConfig& config) const override;
};

}  // namespace ov::intel_cpu::aarch64