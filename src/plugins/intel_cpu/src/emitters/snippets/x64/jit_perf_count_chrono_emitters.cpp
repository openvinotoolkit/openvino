// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef SNIPPETS_DEBUG_CAPS

#    include "jit_perf_count_chrono_emitters.hpp"

#    include "emitters/plugin/x64/utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;
using namespace Xbyak::util;

namespace ov::intel_cpu {

jit_perf_count_chrono_start_emitter::jit_perf_count_chrono_start_emitter(
    dnnl::impl::cpu::x64::jit_generator* host,
    dnnl::impl::cpu::x64::cpu_isa_t host_isa,
    const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_binary_call_emitter(host, host_isa, expr->get_live_regs()) {
    m_start_node = ov::as_type_ptr<snippets::op::PerfCountBegin>(expr->get_node());
}

size_t jit_perf_count_chrono_start_emitter::get_inputs_num() const {
    return 0;
}

void jit_perf_count_chrono_start_emitter::set_start_time(snippets::op::PerfCountBegin* start_node) {
    start_node->set_start_time();
}

void jit_perf_count_chrono_start_emitter::emit_impl(const std::vector<size_t>& in_idxs,
                                                    const std::vector<size_t>& out_idxs) const {
    init_binary_call_regs(0, {});
    const Xbyak::Reg64& aux_reg = get_call_address_reg();
    const Xbyak::Reg64& callee_saved_reg = get_callee_saved_reg();

    EmitABIRegSpills spill(h);
    spill.preamble(get_regs_to_spill());

    const auto& set_start_time_overload = static_cast<void (*)(snippets::op::PerfCountBegin*)>(set_start_time);
    h->mov(aux_reg, reinterpret_cast<size_t>(set_start_time_overload));
    h->mov(abi_param1, reinterpret_cast<size_t>(m_start_node.get()));

    spill.rsp_align(callee_saved_reg.getIdx());
    h->call(aux_reg);
    spill.rsp_restore();

    spill.postamble();
}

///////////////////jit_perf_count_chrono_end_emitter////////////////////////////////////
jit_perf_count_chrono_end_emitter::jit_perf_count_chrono_end_emitter(dnnl::impl::cpu::x64::jit_generator* host,
                                                                     dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                                                                     const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_binary_call_emitter(host, host_isa, expr->get_live_regs()) {
    m_end_node = ov::as_type_ptr<snippets::op::PerfCountEnd>(expr->get_node());
}

size_t jit_perf_count_chrono_end_emitter::get_inputs_num() const {
    return 0;
}

void jit_perf_count_chrono_end_emitter::set_accumulated_time(snippets::op::PerfCountEnd* end_node) {
    end_node->set_accumulated_time();
}

void jit_perf_count_chrono_end_emitter::emit_impl(const std::vector<size_t>& in_idxs,
                                                  const std::vector<size_t>& out_idxs) const {
    init_binary_call_regs(0, {});
    const Xbyak::Reg64& aux_reg = get_call_address_reg();
    const Xbyak::Reg64& callee_saved_reg = get_callee_saved_reg();

    EmitABIRegSpills spill(h);
    spill.preamble(get_regs_to_spill());

    const auto& set_accumulated_time_overload =
        static_cast<void (*)(snippets::op::PerfCountEnd*)>(set_accumulated_time);
    h->mov(aux_reg, reinterpret_cast<size_t>(set_accumulated_time_overload));
    h->mov(abi_param1, reinterpret_cast<size_t>(m_end_node.get()));

    spill.rsp_align(callee_saved_reg.getIdx());
    h->call(aux_reg);
    spill.rsp_restore();

    spill.postamble();
}

}  // namespace ov::intel_cpu

#endif  // SNIPPETS_DEBUG_CAPS
