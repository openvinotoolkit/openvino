// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "emitters/snippets/aarch64/kernel_executors/gemm.hpp"
#include "emitters/snippets/brgemm_generic.hpp"

namespace ov::intel_cpu::aarch64 {

class jit_gemm_emitter : public jit_emitter {
public:
    jit_gemm_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                     dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                     const ov::snippets::lowered::ExpressionPtr& expr,
                     const snippets::KernelExecutorTablePtr& kernel_table);

    size_t get_inputs_count() const override {
        return 2;
    }

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

protected:
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    static const uintptr_t get_execute_function_ptr();
    const uintptr_t get_compiled_kernel_ptr() const;

    std::shared_ptr<GemmKaiKernelExecutor> m_kernel_executor_kai = nullptr;
};

}  // namespace ov::intel_cpu::aarch64
