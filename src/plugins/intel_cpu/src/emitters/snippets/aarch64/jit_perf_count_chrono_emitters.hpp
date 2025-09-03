// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <memory>
#include <vector>

#include "snippets/lowered/expression.hpp"

#ifdef SNIPPETS_DEBUG_CAPS

#    include "jit_binary_call_emitter.hpp"
#    include "snippets/op/perf_count.hpp"

namespace ov::intel_cpu::aarch64 {

class jit_perf_count_chrono_start_emitter : public jit_binary_call_emitter {
public:
    jit_perf_count_chrono_start_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                        dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                        const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {
        return 0;
    }
    size_t get_aux_gprs_count() const override {
        return 1;
    }

private:
    void emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const override;

    static void set_start_time(ov::snippets::op::PerfCountBegin* start_node);

    std::shared_ptr<ov::snippets::op::PerfCountBegin> m_start_node;
};

class jit_perf_count_chrono_end_emitter : public jit_binary_call_emitter {
public:
    jit_perf_count_chrono_end_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                      dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                      const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {
        return 0;
    }
    size_t get_aux_gprs_count() const override {
        return 1;
    }

private:
    void emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const override;

    static void set_accumulated_time(ov::snippets::op::PerfCountEnd* end_node);

    std::shared_ptr<ov::snippets::op::PerfCountEnd> m_end_node;
};

}  // namespace ov::intel_cpu::aarch64

#endif  // SNIPPETS_DEBUG_CAPS
