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

#include "common/utils.hpp"
#include "cpu_memory.h"
#include "emitters/snippets/brgemm_generic.hpp"
#include "emitters/utils.hpp"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p_interface.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x16p32x1b_x16_x16_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x32p16x1b_x32_x32_neon.h"

namespace ov::intel_cpu::aarch64 {

struct GemmCopyBKernelKaiConfig : public snippets::KernelExecutorBase::GenericConfig {
public:
    GemmCopyBKernelKaiConfig() = default;

    bool operator==(const GemmCopyBKernelKaiConfig& rhs) const;
    bool operator!=(const GemmCopyBKernelKaiConfig& rhs) const {
        return !(*this == rhs);
    }

    [[nodiscard]] std::unique_ptr<GenericConfig> get_clone_ptr() const override {
        return std::make_unique<GemmCopyBKernelKaiConfig>(*this);
    }

    [[nodiscard]] bool is_completed() const override;
    [[nodiscard]] bool is_empty() const;

#ifdef SNIPPETS_DEBUG_CAPS
    [[nodiscard]] std::string to_string() const override;
#endif

    void update(size_t N, size_t K, size_t row_stride_bytes, size_t col_stride_bytes);

    [[nodiscard]] size_t hash() const override {
        return m_hash;
    }

    [[nodiscard]] size_t get_N() const {
        return m_N;
    }
    [[nodiscard]] size_t get_K() const {
        return m_K;
    }
    [[nodiscard]] size_t get_copy_b_wei_stride() const {
        return m_copy_b_wei_stride;
    }
    [[nodiscard]] size_t get_copy_b_col_stride() const {
        return m_copy_b_col_stride;
    }
    [[nodiscard]] static size_t get_N_blk() {
        return m_N_blk;
    }

private:
    [[nodiscard]] size_t compute_hash() const;

    // Default value N_blk for iterated repacking.
    // This value doesn't depend on blocking sizes of GemmCPU.
    static constexpr size_t m_N_blk = 64;

    size_t m_N = 0;
    size_t m_K = 0;
    size_t m_copy_b_wei_stride = 0;
    size_t m_copy_b_col_stride = 0;
    size_t m_hash{SIZE_MAX};
};

struct GemmCopyBCompiledKernelF32 {
    std::shared_ptr<kai_matmul_clamp_f32_f32_f32p_ukernel> copy_b_ukernel =
        std::make_shared<kai_matmul_clamp_f32_f32_f32p_ukernel>(ukernel);

    static constexpr kai_matmul_clamp_f32_f32_f32p_ukernel ukernel{
        kai_get_m_step_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla,
        kai_get_n_step_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla,
        kai_get_nr_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla,
        kai_get_kr_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla,
        kai_get_sr_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla,
        kai_get_lhs_offset_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla,
        kai_get_rhs_packed_offset_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla,
        kai_get_dst_offset_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla,
        kai_get_dst_size_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla,
        kai_run_matmul_clamp_f32_f32_f32p16x1b_6x16_neon_mla};
};

struct GemmCopyBCompiledKernelF16 {
    std::shared_ptr<kai_matmul_clamp_f16_f16_f16p_ukernel> copy_b_ukernel =
        std::make_shared<kai_matmul_clamp_f16_f16_f16p_ukernel>(ukernel);

    static constexpr kai_matmul_clamp_f16_f16_f16p_ukernel ukernel{
        kai_get_m_step_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla,
        kai_get_n_step_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla,
        kai_get_nr_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla,
        kai_get_kr_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla,
        kai_get_sr_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla,
        kai_get_lhs_offset_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla,
        kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla,
        kai_get_dst_offset_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla,
        kai_get_dst_size_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla,
        kai_run_matmul_clamp_f16_f16_f16p32x1b_6x32_neon_mla};
};

class GemmCopyBKaiKernelExecutorBase {
protected:
    GemmCopyBKaiKernelExecutorBase() = default;
    ~GemmCopyBKaiKernelExecutorBase() = default;

    static void update_config_common(const ov::snippets::lowered::ExpressionPtr& expr,
                                     const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                     GemmCopyBKernelKaiConfig& config);

    template <typename CompiledKernelT>
    static void ensure_kernel(std::shared_ptr<CompiledKernelT>& kernel);
};

class GemmCopyBF32KaiKernelExecutor
    : public GemmCopyBKaiKernelExecutorBase,
      public snippets::KernelExecutor<GemmCopyBKernelKaiConfig, GemmCopyBCompiledKernelF32> {
public:
    GemmCopyBF32KaiKernelExecutor(GemmCopyBKernelKaiConfig config);
    void update_kernel(const GemmCopyBKernelKaiConfig& config,
                       std::shared_ptr<GemmCopyBCompiledKernelF32>& kernel) const override final;
    static void execute(const GemmCopyBF32KaiKernelExecutor* executor, void* in0, void* out0);

private:
    void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                       const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                       GemmCopyBKernelKaiConfig& config) const override;
};

class GemmCopyBF16KaiKernelExecutor
    : public GemmCopyBKaiKernelExecutorBase,
      public snippets::KernelExecutor<GemmCopyBKernelKaiConfig, GemmCopyBCompiledKernelF16> {
public:
    GemmCopyBF16KaiKernelExecutor(GemmCopyBKernelKaiConfig config);
    void update_kernel(const GemmCopyBKernelKaiConfig& config,
                       std::shared_ptr<GemmCopyBCompiledKernelF16>& kernel) const override final;
    static void execute(const GemmCopyBF16KaiKernelExecutor* executor, void* in0, void* out0);

private:
    void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                       const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                       GemmCopyBKernelKaiConfig& config) const override;
};

}  // namespace ov::intel_cpu::aarch64
