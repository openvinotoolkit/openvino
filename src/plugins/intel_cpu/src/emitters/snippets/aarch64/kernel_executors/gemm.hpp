// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/utils.hpp"
#include "cpu_memory.h"
#include "emitters/snippets/brgemm_generic.hpp"
#include "emitters/utils.hpp"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p8x1biasf32_6x8x4_neon_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p8x1biasf32_f32_f32_neon.h"

namespace ov::intel_cpu::aarch64 {

struct GemmKernelKaiConfig : public BrgemmGenericKernelConfig {
public:
    GemmKernelKaiConfig() = default;

    std::unique_ptr<snippets::KernelExecutorBase::GenericConfig> get_clone_ptr() const override {
        return std::make_unique<GemmKernelKaiConfig>(*this);
    }

    size_t hash() const override {
        return m_hash;
    }

private:
    size_t m_hash{SIZE_MAX};
};

class GemmKaiKernelExecutor
    : public snippets::KernelExecutor<GemmKernelKaiConfig, kai_matmul_clamp_f32_f32_f32p_ukernel> {
public:
    GemmKaiKernelExecutor(GemmKernelKaiConfig config);

    // No need kernel update, just update config is enough for update. The universal ukernel is reused with any config.
    void update_kernel(const GemmKernelKaiConfig& config,
                       std::shared_ptr<kai_matmul_clamp_f32_f32_f32p_ukernel>& kernel) const override final {
        kernel = std::make_shared<kai_matmul_clamp_f32_f32_f32p_ukernel>(ukernel);
    }

    // Function that will be called in runtime to execute the kernel
    static void execute(const GemmKaiKernelExecutor* executor, void* in0, void* in1, void* out0);

private:
    void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                       const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                       GemmKernelKaiConfig& config) const override;

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

}  // namespace ov::intel_cpu::aarch64