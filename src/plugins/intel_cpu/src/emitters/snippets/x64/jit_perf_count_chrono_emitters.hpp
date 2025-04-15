// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef SNIPPETS_DEBUG_CAPS

#    pragma once

#    include "jit_binary_call_emitter.hpp"
#    include "snippets/op/perf_count.hpp"

namespace ov::intel_cpu {

class jit_perf_count_chrono_start_emitter : public jit_binary_call_emitter {
public:
    jit_perf_count_chrono_start_emitter(dnnl::impl::cpu::x64::jit_generator* host,
                                        dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                                        const ov::snippets::lowered::ExpressionPtr& expr);
    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const override;
    static void set_start_time(snippets::op::PerfCountBegin* start_node);

    std::shared_ptr<snippets::op::PerfCountBegin> m_start_node = nullptr;
};

class jit_perf_count_chrono_end_emitter : public jit_binary_call_emitter {
public:
    jit_perf_count_chrono_end_emitter(dnnl::impl::cpu::x64::jit_generator* host,
                                      dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                                      const ov::snippets::lowered::ExpressionPtr& expr);
    size_t get_inputs_num() const override;

private:
    void emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const override;
    static void set_accumulated_time(snippets::op::PerfCountEnd* end_node);

    std::shared_ptr<snippets::op::PerfCountEnd> m_end_node = nullptr;
};

}  // namespace ov::intel_cpu

#endif  // SNIPPETS_DEBUG_CAPS
