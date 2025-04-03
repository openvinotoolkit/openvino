// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef SNIPPETS_DEBUG_CAPS

#    include "jit_perf_count_chrono_emitters.hpp"

#    include "cpu/aarch64/jit_generator.hpp"
#    include "cpu/aarch64/xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_adr.h"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace Xbyak_aarch64;

namespace ov::intel_cpu::aarch64 {

jit_perf_count_chrono_start_emitter::jit_perf_count_chrono_start_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                                                         dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                                                         const std::shared_ptr<ov::Node>& n)
    : jit_emitter(host, host_isa) {
    m_start_node = ov::as_type_ptr<snippets::op::PerfCountBegin>(n);
}

size_t jit_perf_count_chrono_start_emitter::get_inputs_count() const {
    return 0;
}

void jit_perf_count_chrono_start_emitter::set_start_time(snippets::op::PerfCountBegin* start_node) {
    start_node->set_start_time();
}

void jit_perf_count_chrono_start_emitter::emit_impl(const std::vector<size_t>& in_idxs,
                                                    const std::vector<size_t>& out_idxs) const {
    std::unordered_set<size_t> exclude = {};
    store_context(exclude);

    const auto& set_start_time_overload = static_cast<void (*)(snippets::op::PerfCountBegin*)>(set_start_time);
    Xbyak_aarch64::XReg func_reg(9);
    h->mov(func_reg, reinterpret_cast<size_t>(set_start_time_overload));
    Xbyak_aarch64::XReg x0(0);
    h->mov(x0, reinterpret_cast<size_t>(m_start_node.get()));
    h->blr(func_reg);

    restore_context(exclude);
}

///////////////////jit_perf_count_chrono_end_emitter////////////////////////////////////
jit_perf_count_chrono_end_emitter::jit_perf_count_chrono_end_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                                                     const std::shared_ptr<ov::Node>& n)
    : jit_emitter(host, host_isa) {
    m_end_node = ov::as_type_ptr<snippets::op::PerfCountEnd>(n);
}

size_t jit_perf_count_chrono_end_emitter::get_inputs_count() const {
    return 0;
}

void jit_perf_count_chrono_end_emitter::set_accumulated_time(snippets::op::PerfCountEnd* end_node) {
    end_node->set_accumulated_time();
}

void jit_perf_count_chrono_end_emitter::emit_impl(const std::vector<size_t>& in_idxs,
                                                  const std::vector<size_t>& out_idxs) const {
    std::unordered_set<size_t> exclude = {};
    store_context(exclude);

    const auto& set_accumulated_time_overload =
        static_cast<void (*)(snippets::op::PerfCountEnd*)>(set_accumulated_time);
    Xbyak_aarch64::XReg func_reg(9);
    h->mov(func_reg, reinterpret_cast<size_t>(set_accumulated_time_overload));
    Xbyak_aarch64::XReg x0(0);
    h->mov(x0, reinterpret_cast<size_t>(m_end_node.get()));
    h->blr(func_reg);

    restore_context(exclude);
}

}  // namespace ov::intel_cpu::aarch64

#endif  // SNIPPETS_DEBUG_CAPS
