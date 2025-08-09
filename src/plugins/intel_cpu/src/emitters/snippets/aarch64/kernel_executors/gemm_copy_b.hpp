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

struct GemmCopyBKernelKaiConfig : public snippets::KernelExecutorBase::GenericConfig {
public:
    GemmCopyBKernelKaiConfig() = default;
    GemmCopyBKernelKaiConfig(size_t n_blk_size);

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

    void update(size_t N, size_t K, size_t stride);

    [[nodiscard]] size_t hash() const override {
        return m_hash;
    }

    [[nodiscard]] size_t get_N() const {
        return m_N;
    }
    [[nodiscard]] size_t get_K() const {
        return m_K;
    }
    [[nodiscard]] size_t get_stride() const {
        return m_stride;
    }
    [[nodiscard]] size_t get_n_blk_size() const {
        return m_static_params->wei_N_blk;
    }

private:
    struct StaticParams {
        StaticParams(size_t wei_n_blk);

        const size_t wei_N_blk{0};
        const size_t hash{0};

        bool operator==(const StaticParams& rhs) const;
        bool operator!=(const StaticParams& rhs) const {
            return !(*this == rhs);
        }

#ifdef SNIPPETS_DEBUG_CAPS
        [[nodiscard]] std::string to_string() const;
#endif

    private:
        static size_t init_hash(size_t wei_n_blk);
    };

    [[nodiscard]] size_t compute_hash() const;

    std::shared_ptr<StaticParams> m_static_params;
    size_t m_N = 0;
    size_t m_K = 0;
    size_t m_stride = 0;
    size_t m_hash{SIZE_MAX};
};

struct GemmCopyBCompiledKernel {
    std::shared_ptr<kai_matmul_clamp_f32_f32_f32p_ukernel> copy_b_ukernel =
        std::make_shared<kai_matmul_clamp_f32_f32_f32p_ukernel>(ukernel);
    std::shared_ptr<std::vector<uint8_t>> bias_buffer = std::make_shared<std::vector<uint8_t>>();

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

class GemmCopyBKaiKernelExecutor : public snippets::KernelExecutor<GemmCopyBKernelKaiConfig, GemmCopyBCompiledKernel> {
public:
    GemmCopyBKaiKernelExecutor(GemmCopyBKernelKaiConfig config);

    void update_kernel(const GemmCopyBKernelKaiConfig& config,
                       std::shared_ptr<GemmCopyBCompiledKernel>& kernel) const override final;

    // Function that will be called in runtime to execute the kernel
    static void execute(const GemmCopyBKaiKernelExecutor* executor, void* in0, void* out0);

private:
    void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                       const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                       GemmCopyBKernelKaiConfig& config) const override;
};

}  // namespace ov::intel_cpu::aarch64
