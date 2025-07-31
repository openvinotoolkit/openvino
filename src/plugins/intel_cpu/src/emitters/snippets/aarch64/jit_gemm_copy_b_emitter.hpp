// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "emitters/snippets/aarch64/kernel_executors/gemm_copy_b.hpp"

namespace ov::intel_cpu::aarch64 {
class jit_gemm_copy_b_emitter : public jit_emitter {
public:
    jit_gemm_copy_b_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                            dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                            const ov::snippets::lowered::ExpressionPtr& expr,
                            const snippets::KernelExecutorTablePtr& kernel_table);

    size_t get_inputs_count() const override {
        return 1;
    }

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

protected:
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    static uintptr_t get_execute_function_ptr();
    uintptr_t get_compiled_kernel_ptr() const;

    std::shared_ptr<GemmCopyBKaiKernelExecutor> m_kernel_executor = nullptr;
};

}  // namespace ov::intel_cpu::aarch64