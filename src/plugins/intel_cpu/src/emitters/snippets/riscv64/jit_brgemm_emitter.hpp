// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <vector>

#include "cache/multi_cache.h"
#include "emitters/snippets/riscv64/jit_binary_call_emitter.hpp"
#include "emitters/snippets/riscv64/kernel_executors/brgemm.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"

namespace ov::intel_cpu::riscv64 {

class jit_brgemm_emitter : public jit_binary_call_emitter {
public:
    jit_brgemm_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                       ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                       const ov::snippets::lowered::ExpressionPtr& expr,
                       const snippets::KernelExecutorTablePtr& kernel_table,
                       const ov::intel_cpu::MultiCacheWeakPtr& compiled_kernel_cache);

    size_t get_inputs_num() const override {
        return 2;
    }

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    std::vector<size_t> m_memory_offsets;
    std::vector<size_t> m_buffer_ids;
    std::shared_ptr<BrgemmKernelExecutor> m_kernel_executor;
};

}  // namespace ov::intel_cpu::riscv64
