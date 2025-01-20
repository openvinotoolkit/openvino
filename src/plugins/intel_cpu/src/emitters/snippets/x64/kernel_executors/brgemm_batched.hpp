// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "brgemm_base.hpp"

namespace ov::intel_cpu {

struct BrgemmBatchedKernelConfig : public x64::BrgemmBaseKernelConfig {
public:
    BrgemmBatchedKernelConfig(const element::Type& in0_dtype,
                              const element::Type& in1_dtype,
                              size_t iter_count,
                              bool is_with_comp,
                              dnnl::impl::cpu::x64::cpu_isa_t primitive_isa);
    BrgemmBatchedKernelConfig() = delete;

    std::unique_ptr<snippets::KernelExecutorBase::GenericConfig> get_clone_ptr() const override {
        return std::unique_ptr<BrgemmBatchedKernelConfig>(new BrgemmBatchedKernelConfig(*this));
    }

    bool is_with_comp() const {
        return m_static_params->is_with_comp;
    }

    size_t get_iter_count() const {
        return m_iter_count;
    }

private:
    struct StaticParams : StaticBaseParams {
        StaticParams(const element::Type& in0_dtype,
                     const element::Type& in1_dtype,
                     bool is_with_comp,
                     dnnl::impl::cpu::x64::cpu_isa_t primitive_isa);

        const bool is_with_comp{false};

        bool operator==(const StaticParams& rhs) const;
        bool operator!=(const StaticParams& rhs) const {
            return !(*this == rhs);
        }
#ifdef SNIPPETS_DEBUG_CAPS
        std::string to_string() const override;
#endif
    private:
        static size_t compute_hash(bool is_with_comp);
    };

    std::shared_ptr<StaticBaseParams> get_static_params() const override {
        return m_static_params;
    }

    std::shared_ptr<StaticParams> m_static_params{nullptr};
    size_t m_iter_count{1};
};

// The `update_kernel` method verifies that a compiled kernel is not nullptr.
// However, the compiled kernel might be empty in cases if nothing is to be compiled (`Config.is_empty() == true`).
// To cover this case, we wrap the `brgemm_kernel_t` in the separate structure which may contain empty `brgemm_kernel_t`
struct BrgemmBatchedCompiledKernel {
    std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> brgemm_kernel = nullptr;
};

class BrgemmBatchedKernelExecutor : public x64::BrgemmBaseKernelExecutor,
                                    public CPUKernelExecutor<BrgemmBatchedKernelConfig, BrgemmBatchedCompiledKernel> {
public:
    struct call_args {
        const void* A = nullptr;
        const void* B = nullptr;
        void* C = nullptr;
        void* scratch = nullptr;
    };
    BrgemmBatchedKernelExecutor(ov::intel_cpu::MultiCacheWeakPtr kernel_cache, BrgemmBatchedKernelConfig config);

    /** Function that will be called in runtime to execute the  */
    static void execute(const BrgemmBatchedKernelExecutor* execkernelutor, call_args* args);

protected:
    std::shared_ptr<BrgemmBatchedCompiledKernel> compile_kernel(const BrgemmBatchedKernelConfig& c) const override;

    void update_config(const ov::snippets::lowered::ExpressionPtr& expr,
                       const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                       BrgemmBatchedKernelConfig& config) const override;

    static void execute_brgemm(const std::shared_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& kernel,
                               size_t bs,
                               size_t stride_A,
                               size_t stride_B,
                               const void* src,
                               const void* wei,
                               void* dst,
                               void* scratch,
                               bool with_comp);
};

}  // namespace ov::intel_cpu
