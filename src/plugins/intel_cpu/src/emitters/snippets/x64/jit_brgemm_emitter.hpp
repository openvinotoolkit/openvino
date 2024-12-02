// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/snippets/x64/kernel_executors/brgemm.hpp"

namespace ov {
namespace intel_cpu {

class jit_brgemm_emitter : public jit_emitter {
public:
    jit_brgemm_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                       const ov::snippets::lowered::ExpressionPtr& expr,
                       const snippets::KernelExecutorTablePtr& kernel_table,
                       const ov::intel_cpu::MultiCacheWeakPtr& compiled_kernel_cache);

    size_t get_inputs_num() const override { return m_memory_offsets.size() - 1; }
    static std::set<std::vector<element::Type>> get_supported_precisions(const std::shared_ptr<ov::Node>& node = nullptr);

private:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    // Note: offsets order: A, B, C (+ scratchpad, if needed). Values can be dynamic_value if offset is calculated in runtime
    std::vector<size_t> m_memory_offsets{};
    // Note: cluster ids order: A, B, C (+ scratchpad, if needed). Values can be dynamic_value if there is no buffer
    std::vector<size_t> m_buffer_ids{};
    std::shared_ptr<BrgemmKernelExecutor> m_kernel_executor = nullptr;

#ifdef SNIPPETS_DEBUG_CAPS
    friend std::string init_info_jit_brgemm_emitter(const jit_brgemm_emitter *emitter);
#endif
};

}   // namespace intel_cpu
}   // namespace ov
