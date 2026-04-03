// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_perf_count_chrono_emitters.hpp"

#ifdef SNIPPETS_DEBUG_CAPS

#    include <cstddef>
#    include <cstdint>
#    include <vector>

#    include "openvino/core/except.hpp"
#    include "openvino/core/type.hpp"
#    include "snippets/lowered/expression.hpp"
#    include "snippets/op/perf_count.hpp"
#    include "xbyak_riscv/xbyak_riscv.hpp"

namespace ov::intel_cpu::riscv64 {

using jit_generator_t = ov::intel_cpu::riscv64::jit_generator_t;
using cpu_isa_t = ov::intel_cpu::riscv64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

jit_perf_count_chrono_start_emitter::jit_perf_count_chrono_start_emitter(jit_generator_t* host,
                                                                         cpu_isa_t host_isa,
                                                                         const ExpressionPtr& expr)
    : jit_binary_call_emitter(host, host_isa, expr->get_live_regs()) {
    m_start_node = ov::as_type_ptr<ov::snippets::op::PerfCountBegin>(expr->get_node());
    OPENVINO_ASSERT(m_start_node, "PerfCountBegin node is null");
}

void jit_perf_count_chrono_start_emitter::set_start_time(ov::snippets::op::PerfCountBegin* start_node) {
    start_node->set_start_time();
}

void jit_perf_count_chrono_start_emitter::emit_impl([[maybe_unused]] const std::vector<size_t>& in_idxs,
                                                    [[maybe_unused]] const std::vector<size_t>& out_idxs) const {
    init_binary_call_regs(/*num_binary_args*/ 1, /*used_gpr_idxs*/ {});
    binary_call_preamble();

    const auto& func_reg = get_call_address_reg();
    const auto fn_ptr =
        reinterpret_cast<size_t>(static_cast<void (*)(ov::snippets::op::PerfCountBegin*)>(set_start_time));
    h->uni_li(func_reg, fn_ptr);
    h->uni_li(Xbyak_riscv::a0, reinterpret_cast<size_t>(m_start_node.get()));
    h->jalr(Xbyak_riscv::ra, func_reg);

    binary_call_postamble();
}

jit_perf_count_chrono_end_emitter::jit_perf_count_chrono_end_emitter(jit_generator_t* host,
                                                                     cpu_isa_t host_isa,
                                                                     const ExpressionPtr& expr)
    : jit_binary_call_emitter(host, host_isa, expr->get_live_regs()) {
    m_end_node = ov::as_type_ptr<ov::snippets::op::PerfCountEnd>(expr->get_node());
    OPENVINO_ASSERT(m_end_node, "PerfCountEnd node is null");
}

void jit_perf_count_chrono_end_emitter::set_accumulated_time(ov::snippets::op::PerfCountEnd* end_node) {
    end_node->set_accumulated_time();
}

void jit_perf_count_chrono_end_emitter::emit_impl([[maybe_unused]] const std::vector<size_t>& in_idxs,
                                                  [[maybe_unused]] const std::vector<size_t>& out_idxs) const {
    init_binary_call_regs(/*num_binary_args*/ 1, /*used_gpr_idxs*/ {});
    binary_call_preamble();

    const auto& func_reg = get_call_address_reg();
    const auto fn_ptr =
        reinterpret_cast<size_t>(static_cast<void (*)(ov::snippets::op::PerfCountEnd*)>(set_accumulated_time));
    h->uni_li(func_reg, fn_ptr);
    h->uni_li(Xbyak_riscv::a0, reinterpret_cast<size_t>(m_end_node.get()));
    h->jalr(Xbyak_riscv::ra, func_reg);

    binary_call_postamble();
}

}  // namespace ov::intel_cpu::riscv64

#endif  // SNIPPETS_DEBUG_CAPS
