// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "emitters/tpp/aarch64/kernel_executors/kleidiai_gemm.hpp"
#include "emitters/tpp/common/kernel_executors/brgemm.hpp"

// comment out to use libxsmm backend
#define KLEIDIAI_BACKEDN

namespace ov::intel_cpu::aarch64 {

class jit_brgemm_emitter : public jit_emitter {
public:
    jit_brgemm_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                       dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                       const ov::snippets::lowered::ExpressionPtr& expr,
                       const snippets::KernelExecutorTablePtr& kernel_table,
                       const ov::intel_cpu::MultiCacheWeakPtr& compiled_kernel_cache);

    size_t get_inputs_count() const override {
        return 2;
    }

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

protected:
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_code_impl(const std::vector<size_t>& in_idxs,
                        const std::vector<size_t>& out_idxs,
                        const std::vector<size_t>& pool_vec_idxs = {},
                        const std::vector<size_t>& pool_gpr_idxs = {}) const override;

    const uintptr_t get_execute_function_ptr() const;
    const uintptr_t get_compiled_kernel_ptr() const;

#ifdef KLEIDIAI_BACKEDN
    std::shared_ptr<ov::intel_cpu::tpp::BrgemmKaiKernelExecutor> m_kernel_executor = nullptr;
#else
    std::shared_ptr<ov::intel_cpu::tpp::BrgemmKernelExecutor> m_kernel_executor = nullptr;
#endif
};

}  // namespace ov::intel_cpu::aarch64
