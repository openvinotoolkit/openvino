// Copyright (C) 2020-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/snippets/x64/kernel_executors/brgemm_base.hpp"
#include "libxsmm.h"
#include "emitters/tpp/x64/utils.hpp"

namespace ov {
namespace intel_cpu {

struct BrgemmTppKernelConfig : public BrgemmBaseKernelConfig {
public:
    BrgemmTppKernelConfig(const element::Type& in0_dtype,
                          const element::Type& in1_dtype,
                          libxsmm_bitfield compile_flags,
                          bool m_prefetching_flags,
                          dnnl::impl::cpu::x64::cpu_isa_t primitive_isa = dnnl::impl::cpu::x64::cpu_isa_t::isa_undef);
    BrgemmTppKernelConfig() = delete;

    std::unique_ptr<snippets::KernelExecutorBase::GenericConfig> get_clone_ptr() const override {
        return std::unique_ptr<BrgemmTppKernelConfig>(new BrgemmTppKernelConfig(*this));
    }

    libxsmm_bitfield get_compile_flags() const {
        return m_compile_flags;
    }
    bool get_prefetching_flags() const {
        return m_static_params->m_prefetching_flags;
    }
    libxsmm_datatype get_type_in0() const {
        return m_static_params->m_type_in0;
    }
    libxsmm_datatype get_type_in1() const {
        return m_static_params->m_type_in1;
    }
    libxsmm_datatype get_type_out0() const {
        return m_static_params->m_type_out0;
    }
    libxsmm_datatype get_type_exec() const {
        return m_static_params->m_type_exec;
    }

private:
    struct StaticParams : StaticBaseParams {
        StaticParams(const element::Type& in0_dtype,
                     const element::Type& in1_dtype,
                     bool prefetching_flags,
                     dnnl::impl::cpu::x64::cpu_isa_t primitive_isa);

        bool operator==(const StaticParams& rhs) const;
        bool operator!=(const StaticParams& rhs) const {
            return !(*this == rhs);
        }
        libxsmm_datatype m_type_in0;
        libxsmm_datatype m_type_in1;
        libxsmm_datatype m_type_out0;
        libxsmm_datatype m_type_exec;
        libxsmm_bitfield m_compile_flags {0};
        const bool m_prefetching_flags{false};
#ifdef SNIPPETS_DEBUG_CAPS
        std::string to_string() const;
#endif
    private:
        static size_t compute_hash(bool prefetching_flags);
    };

    std::shared_ptr<StaticBaseParams> get_static_params() const override {
        return m_static_params;
    }

    libxsmm_bitfield m_compile_flags{0};

    std::shared_ptr<StaticParams> m_static_params{nullptr};
};

// The `update_kernel` method verifies that a compiled kernel is not nullptr.
// However, the compiled kernel might be empty in cases if nothing is to be compiled (`Config.is_empty() == true`).
// To cover this case, we wrap the `libxsmm_gemmfunction` in the separate structure which may contain empty
// `libxsmm_gemmfunction`
struct BrgemmTppCompiledKernel {
    std::shared_ptr<libxsmm_gemmfunction> brgemm_kernel = nullptr;
};

class BrgemmTppKernelExecutor : public BrgemmBaseKernelExecutor,
                                public CPUKernelExecutor<BrgemmTppKernelConfig, BrgemmTppCompiledKernel> {
public:
    BrgemmTppKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmTppKernelConfig config);

    /** Function that will be called in runtime to execute the kernel */
    static void execute(const BrgemmTppKernelExecutor* executor, void* in0, void* in1, void* out0);

protected:
    std::shared_ptr<BrgemmTppCompiledKernel> compile_kernel(const BrgemmTppKernelConfig& c) const override;

    void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                       const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                       BrgemmTppKernelConfig& config) const override;
};
#define GET_OFF_BRGEMM_ARGS(field) offsetof(BrgemmKernelExecutor::call_args, field)

}  // namespace intel_cpu
}  // namespace ov
