// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl_common_types.h>

#include <cpu/x64/brgemm/brgemm_types.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstddef>
#include <memory>
#include <string>

#include "brgemm_base.hpp"
#include "cache/multi_cache.h"
#include "emitters/snippets/cpu_kernel_executor_table.hpp"
#include "emitters/utils.hpp"
#include "nodes/kernels/x64/jit_online_softmax_kernel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"
#include "transformations/snippets/x64/op/fa_utils.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::x64 {

struct FAKernelConfig : public snippets::KernelExecutorBase::GenericConfig {
public:
    FAKernelConfig() = default;
    FAKernelConfig(const ov::intel_cpu::fa_utils::FAConfig& fa_config);

    bool operator==(const FAKernelConfig& rhs) const;
    bool operator!=(const FAKernelConfig& rhs) const {
        return !(*this == rhs);
    }

    [[nodiscard]] bool is_completed() const override;
    [[nodiscard]] bool is_empty() const;

#ifdef SNIPPETS_DEBUG_CAPS
    [[nodiscard]] std::string to_string() const override;
#endif

    void update(dnnl_dim_t q_len, dnnl_dim_t kv_len, dnnl_dim_t head_size_1, dnnl_dim_t head_size_2);

    [[nodiscard]] std::unique_ptr<snippets::KernelExecutorBase::GenericConfig> get_clone_ptr() const override {
        return std::make_unique<FAKernelConfig>(*this);
    }

    [[nodiscard]] size_t get_q_seq_len() const {
        return m_q_seq_len;
    }
    [[nodiscard]] size_t get_kv_seq_len() const {
        return m_kv_seq_len;
    }
    [[nodiscard]] size_t get_qk_head_size() const {
        return m_qk_head_size;
    }
    [[nodiscard]] size_t get_v_head_size() const {
        return m_v_head_size;
    }
    [[nodiscard]] size_t hash() const override {
        return m_hash;
    }

    // get static params
    [[nodiscard]] dnnl_data_type_t get_src_dt() const {
        return m_static_params->m_src_dt;
    }
    [[nodiscard]] dnnl_data_type_t get_wei_dt() const {
        return m_static_params->m_wei_dt;
    }
    [[nodiscard]] dnnl_data_type_t get_original_wei_dt() const {
        return m_static_params->m_original_wei_dt;
    }
    [[nodiscard]] dnnl::impl::cpu::x64::cpu_isa_t get_isa() const {
        return m_static_params->m_isa;
    }
    [[nodiscard]] bool is_transposed_B() const {
        return m_static_params->m_is_transposed_B;
    }
    [[nodiscard]] size_t get_q_len_blk() const {
        return m_static_params->m_q_len_blk;
    }
    [[nodiscard]] size_t get_kv_len_blk() const {
        return m_static_params->m_kv_len_blk;
    }

private:
    struct StaticParams {
        StaticParams(const element::Type& src_type,
                     const element::Type& wei_type,
                     const element::Type& original_wei_type,
                     dnnl::impl::cpu::x64::cpu_isa_t isa,
                     bool is_transposed_B,
                     dnnl_dim_t q_len_blk,
                     dnnl_dim_t kv_len_blk);

        const dnnl_data_type_t m_src_dt{dnnl_data_type_undef}, m_wei_dt{dnnl_data_type_undef};
        const dnnl_data_type_t m_original_wei_dt{dnnl_data_type_undef};
        const dnnl::impl::cpu::x64::cpu_isa_t m_isa{dnnl::impl::cpu::x64::isa_undef};
        const bool m_is_transposed_B{false};
        const dnnl_dim_t m_q_len_blk{0}, m_kv_len_blk{0};
        const size_t m_hash{0};

        bool operator==(const StaticParams& rhs) const;
        bool operator!=(const StaticParams& rhs) const {
            return !(*this == rhs);
        }

#ifdef SNIPPETS_DEBUG_CAPS
        [[nodiscard]] std::string to_string() const;
#endif

    private:
        static size_t init_hash(const dnnl_data_type_t& src_dt,
                                const dnnl_data_type_t& wei_dt,
                                const dnnl_data_type_t& original_wei_dt,
                                dnnl::impl::cpu::x64::cpu_isa_t primitive_isa,
                                bool is_transposed_B,
                                dnnl_dim_t q_len_blk,
                                dnnl_dim_t kv_len_blk);
    };

    [[nodiscard]] size_t compute_hash() const;

    std::shared_ptr<StaticParams> m_static_params;
    dnnl_dim_t m_q_seq_len = 0;
    dnnl_dim_t m_kv_seq_len = 0;
    dnnl_dim_t m_qk_head_size = 0;
    dnnl_dim_t m_v_head_size = 0;
    size_t m_hash{SIZE_MAX};
};

struct FACompiledKernel {
    std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> brgemm_qk_MN_ukernel = nullptr;
    std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> brgemm_qk_mN_ukernel = nullptr;
    std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> brgemm_qk_Mn_ukernel = nullptr;
    std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> brgemm_qk_mn_ukernel = nullptr;
    std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> brgemm_sv_MK_ukernel = nullptr;  // beta is 1
    std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> brgemm_sv_mK_ukernel = nullptr;
    std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> brgemm_sv_Mk_ukernel = nullptr;
    std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> brgemm_sv_mk_ukernel = nullptr;
    std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> brgemm_sv_MK_ukernel_init = nullptr;  // beta is 0
    std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> brgemm_sv_mK_ukernel_init = nullptr;
    std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> brgemm_sv_Mk_ukernel_init = nullptr;
    std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> brgemm_sv_mk_ukernel_init = nullptr;
    std::shared_ptr<jit_uni_online_softmax_kernel> online_softmax_ukernel = nullptr;
    std::shared_ptr<jit_uni_online_softmax_kernel> online_softmax_ukernel_init = nullptr;
    // this buffer include qk result and calibration coefficient.
    std::shared_ptr<std::vector<uint8_t>> buffer = std::make_shared<std::vector<uint8_t>>();
};

class FAKernelExecutor : public BrgemmBaseKernelExecutor, public CPUKernelExecutor<FAKernelConfig, FACompiledKernel> {
public:
    struct call_args {
        const void* A = nullptr;
        const void* B = nullptr;
        const void* C = nullptr;
        void* D = nullptr;
    };
    FAKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, FAKernelConfig config);

    [[nodiscard]] std::shared_ptr<FACompiledKernel> compile_kernel(const FAKernelConfig& c) const override;

    // Function that will be called in runtime to execute the kernel
    static void execute(const FAKernelExecutor* executor, call_args* args);

private:
    void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                       const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                       FAKernelConfig& config) const override;

    std::shared_ptr<jit_uni_online_softmax_kernel> m_online_softmax_ukernel = nullptr;
    std::shared_ptr<jit_uni_online_softmax_kernel> m_online_softmax_ukernel_init = nullptr;
};

}  // namespace ov::intel_cpu::x64