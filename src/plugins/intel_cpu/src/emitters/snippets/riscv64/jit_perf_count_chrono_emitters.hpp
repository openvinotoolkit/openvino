// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "nodes/kernels/riscv64/cpu_isa_traits.hpp"
#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "snippets/lowered/expression.hpp"

#ifdef SNIPPETS_DEBUG_CAPS

#    include "jit_binary_call_emitter.hpp"
#    include "snippets/op/perf_count.hpp"

namespace ov::intel_cpu::riscv64 {

class jit_perf_count_chrono_start_emitter : public jit_binary_call_emitter {
public:
    jit_perf_count_chrono_start_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                        ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                        const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {
        return 0;
    }

private:
    void emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const override;

    static void set_start_time(ov::snippets::op::PerfCountBegin* start_node);

    std::shared_ptr<ov::snippets::op::PerfCountBegin> m_start_node = nullptr;
};

class jit_perf_count_chrono_end_emitter : public jit_binary_call_emitter {
public:
    jit_perf_count_chrono_end_emitter(ov::intel_cpu::riscv64::jit_generator_t* host,
                                      ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                                      const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {
        return 0;
    }

private:
    void emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const override;

    static void set_accumulated_time(ov::snippets::op::PerfCountEnd* end_node);

    std::shared_ptr<ov::snippets::op::PerfCountEnd> m_end_node = nullptr;
};

}  // namespace ov::intel_cpu::riscv64

#endif  // SNIPPETS_DEBUG_CAPS
