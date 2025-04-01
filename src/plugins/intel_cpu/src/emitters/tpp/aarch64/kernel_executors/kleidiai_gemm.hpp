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

namespace ov::intel_cpu::tpp {

struct BrgemmKernelKaiConfig : public BrgemmGenericKernelConfig {
public:
    BrgemmKernelKaiConfig() = default;

    std::unique_ptr<snippets::KernelExecutorBase::GenericConfig> get_clone_ptr() const override {
        return std::unique_ptr<BrgemmKernelKaiConfig>(new BrgemmKernelKaiConfig(*this));
    }

    size_t hash() const override {
        return m_hash;
    }

private:
    size_t m_hash{SIZE_MAX};
};

struct BrgemmTppKaiCompiledKernel {
    std::shared_ptr<kai_matmul_clamp_f32_f32_f32p_ukernel> brgemm_kernel = nullptr;
};

class BrgemmKaiKernelExecutor : public snippets::KernelExecutor<BrgemmKernelKaiConfig, BrgemmTppKaiCompiledKernel> {
public:
    BrgemmKaiKernelExecutor(BrgemmKernelKaiConfig config);

    void update_kernel(const BrgemmKernelKaiConfig& config,
                       std::shared_ptr<BrgemmTppKaiCompiledKernel>& kernel) const override final {}

    // Function that will be called in runtime to execute the kernel
    static void execute(const BrgemmKaiKernelExecutor* executor, void* in0, void* in1, void* out0);
    void* get_packed_mem() const {
        return rhsPackedMem.data();
    }
    void* get_bias_mem() const {
        return biasMem.data();
    }
    mutable size_t rhsPackedSize = 0;
    mutable size_t biasSize = 0;

private:
    void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                       const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                       BrgemmKernelKaiConfig& config) const override;

    mutable std::vector<uint8_t> rhsPackedMem;
    mutable std::vector<uint8_t> biasMem;

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

}  // namespace ov::intel_cpu::tpp