// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "jit_binary_call_emitter.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "emitters/snippets/x64/kernel_executors/fa.hpp"

namespace ov::intel_cpu {

class jit_fa_emitter : public jit_binary_call_emitter {
public:
    jit_fa_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                   dnnl::impl::cpu::x64::cpu_isa_t isa,
                   const ov::snippets::lowered::ExpressionPtr& expr,
                   const snippets::KernelExecutorTablePtr& kernel_table,
                   const ov::intel_cpu::MultiCacheWeakPtr& compiled_kernel_cache);

    size_t get_inputs_num() const override {
        return 3;
    }

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

protected:
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    static const uintptr_t get_execute_function_ptr();
    const uintptr_t get_compiled_kernel_ptr() const;

    std::shared_ptr<ov::intel_cpu::x64::FAKernelExecutor> m_kernel_executor_fa = nullptr;

private:
    std::vector<size_t> m_memory_offsets;
};

}  // namespace ov::intel_cpu