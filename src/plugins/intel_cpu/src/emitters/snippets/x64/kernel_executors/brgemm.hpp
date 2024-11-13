// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "brgemm_base.hpp"

namespace ov {
namespace intel_cpu {

struct BrgemmKernelConfig : public snippets::KernelExecutorBase::GenericConfig, BrgemmBaseKernelConfig {
public:
    BrgemmKernelConfig(const element::Type& in0_dtype, const element::Type& in1_dtype,
                       bool is_with_comp, dnnl::impl::cpu::x64::cpu_isa_t primitive_isa);
    BrgemmKernelConfig() = delete;

    std::unique_ptr<GenericConfig> get_clone_ptr() const override {
        return std::unique_ptr<BrgemmKernelConfig>( new BrgemmKernelConfig(*this));
    }

    bool is_completed() const override { return BrgemmBaseKernelConfig::is_completed(); }
    size_t hash() const override { return BrgemmBaseKernelConfig::hash(); }

    bool is_with_comp() const { return std::static_pointer_cast<StaticParams>(m_static_params)->is_with_comp; }

#ifdef SNIPPETS_DEBUG_CAPS
    std::string to_string() const override { return BrgemmBaseKernelConfig::to_string(); }
#endif

private:
    struct StaticParams : StaticBaseParams {
        StaticParams(const element::Type& in0_dtype, const element::Type& in1_dtype, bool is_with_comp, dnnl::impl::cpu::x64::cpu_isa_t primitive_isa);

        const bool is_with_comp {false};

        size_t hash() const override { return m_hash; }

        bool operator==(const StaticParams& rhs) const;
        bool operator!=(const StaticParams& rhs) const { return !(*this == rhs); }
#ifdef SNIPPETS_DEBUG_CAPS
        std::string to_string() const;
#endif
    private:
        size_t compute_hash() const override;

        const size_t m_hash {0};
    };
};

struct BrgemmCompiledKernel {
    std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> brgemm_kernel = nullptr;
};

class BrgemmKernelExecutor : public BrgemmBaseKernelExecutor,
                             public CPUKernelExecutor<BrgemmKernelConfig, BrgemmCompiledKernel> {
public:
    struct call_args {
        const void* A = nullptr;
        const void* B = nullptr;
        void* C = nullptr;
        void* scratch = nullptr;
    };
    BrgemmKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmKernelConfig config);

    /** Function that will be called in runtime to execute the kernel */
    static void execute(const BrgemmKernelExecutor* executor, call_args* args);

protected:
    std::shared_ptr<BrgemmCompiledKernel> compile_kernel(const BrgemmKernelConfig& c) const override;

    void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                       const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                       BrgemmKernelConfig& config) const override;
};
#define GET_OFF_BRGEMM_ARGS(field) offsetof(BrgemmKernelExecutor::call_args, field)

#ifdef SNIPPETS_DEBUG_CAPS
class BrgemmKernelReferenceExecutor : public BrgemmKernelExecutor {
public:
    BrgemmKernelReferenceExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmKernelConfig config);
    using BrgemmKernelExecutor::execute;
protected:
    std::shared_ptr<BrgemmCompiledKernel> compile_kernel(const BrgemmKernelConfig& c) const override;
};
struct brgemm_ref_kernel : public dnnl::impl::cpu::x64::brgemm_kernel_t {
    brgemm_ref_kernel(BrgemmKernelConfig c);
    void operator()(dnnl::impl::cpu::x64::brgemm_kernel_params_t *) const override;
    dnnl_status_t create_kernel() override { return dnnl_status_t::dnnl_success; }
    const dnnl::impl::cpu::x64::jit_generator *get_jit_generator() const override {
        OV_CPU_JIT_EMITTER_THROW("get_jit_generator should not be called for reference kernel");
        return nullptr;
    }
private:
    BrgemmKernelConfig m_config;
};
#endif
}   // namespace intel_cpu
}   // namespace ov
