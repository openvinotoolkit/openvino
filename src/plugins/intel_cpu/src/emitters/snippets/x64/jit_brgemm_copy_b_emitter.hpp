// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "jit_binary_call_emitter.hpp"
#include "kernel_executors/brgemm_copy_b.hpp"

namespace ov::intel_cpu {

class jit_brgemm_copy_b_emitter : public jit_binary_call_emitter {
public:
    jit_brgemm_copy_b_emitter(dnnl::impl::cpu::x64::jit_generator* h,
                              dnnl::impl::cpu::x64::cpu_isa_t isa,
                              const ov::snippets::lowered::ExpressionPtr& expr,
                              const snippets::KernelExecutorTablePtr& kernel_table,
                              const ov::intel_cpu::MultiCacheWeakPtr& compiled_kernel_cache);
    size_t get_inputs_num() const override {
        return 1;
    }
    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr) {
        return {{element::i8}, {element::bf16}, {element::f16}, {element::f32}};
    }

private:
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    std::vector<size_t> m_memory_offsets{};
    std::vector<size_t> m_buffer_ids{};
    std::shared_ptr<BrgemmCopyBKernelExecutor> m_kernel_executor{nullptr};
    bool m_with_comp{false};

#ifdef SNIPPETS_DEBUG_CAPS
    friend std::string init_info_jit_brgemm_copy_b_emitter(const jit_brgemm_copy_b_emitter* emitter);
#endif
};

}  // namespace ov::intel_cpu
