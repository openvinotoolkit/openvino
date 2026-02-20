// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cache/multi_cache.h"
#include "emitters/snippets/aarch64/jit_binary_call_emitter.hpp"
#include "emitters/snippets/aarch64/kernel_executors/gemm_copy_b.hpp"
#include "snippets/emitter.hpp"

namespace ov::intel_cpu::aarch64 {
class jit_gemm_copy_b_emitter : public jit_binary_call_emitter {
public:
    jit_gemm_copy_b_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                            dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                            const ov::snippets::lowered::ExpressionPtr& expr,
                            const snippets::KernelExecutorTablePtr& kernel_table,
                            const ov::intel_cpu::MultiCacheWeakPtr& compiled_kernel_cache);

    size_t get_inputs_count() const override {
        return 1;
    }

    static std::set<std::vector<element::Type>> get_supported_precisions(
        const std::shared_ptr<ov::Node>& node = nullptr);

protected:
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    template <typename ExecutorT>
    void emit_call(const std::vector<size_t>& mem_ptrs_idxs) const;

    std::shared_ptr<GemmCopyBKaiKernelExecutorBase> m_kernel_executor = nullptr;
    std::vector<size_t> m_memory_offsets;
    std::vector<size_t> m_buffer_ids;
    bool m_is_f16 = false;

#ifdef SNIPPETS_DEBUG_CAPS
    friend std::string init_info_jit_gemm_copy_b_emitter(const jit_gemm_copy_b_emitter* emitter);
#endif
};

}  // namespace ov::intel_cpu::aarch64
