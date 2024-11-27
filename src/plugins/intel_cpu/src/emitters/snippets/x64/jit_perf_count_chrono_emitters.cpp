// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef SNIPPETS_DEBUG_CAPS

#include "jit_perf_count_chrono_emitters.hpp"

#include "emitters/plugin/x64/utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;
using namespace Xbyak::util;

namespace ov {
namespace intel_cpu {

jit_perf_count_chrono_start_emitter::jit_perf_count_chrono_start_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                                            const std::shared_ptr<ov::Node>& n) : jit_emitter(host, host_isa) {
    m_start_node = ov::as_type_ptr<snippets::op::PerfCountBegin>(n);
}

size_t jit_perf_count_chrono_start_emitter::get_inputs_num() const {
    return 0;
}

void jit_perf_count_chrono_start_emitter::set_start_time(snippets::op::PerfCountBegin* start_node) {
    start_node->set_start_time();
}

void jit_perf_count_chrono_start_emitter::emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    EmitABIRegSpills spill(h);
    spill.preamble();

    const auto &set_start_time_overload = static_cast<void (*)(snippets::op::PerfCountBegin*)>(set_start_time);
    h->mov(h->rax, reinterpret_cast<size_t>(set_start_time_overload));
    h->mov(abi_param1, reinterpret_cast<size_t>(m_start_node.get()));

    spill.rsp_align();
    h->call(h->rax);
    spill.rsp_restore();

    spill.postamble();
}

///////////////////jit_perf_count_chrono_end_emitter////////////////////////////////////
jit_perf_count_chrono_end_emitter::jit_perf_count_chrono_end_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
    const std::shared_ptr<ov::Node>& n) : jit_emitter(host, host_isa) {
    m_end_node = ov::as_type_ptr<snippets::op::PerfCountEnd>(n);
}

size_t jit_perf_count_chrono_end_emitter::get_inputs_num() const {
    return 0;
}

void jit_perf_count_chrono_end_emitter::set_accumulated_time(snippets::op::PerfCountEnd* end_node) {
    end_node->set_accumulated_time();
}

void jit_perf_count_chrono_end_emitter::emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    EmitABIRegSpills spill(h);
    spill.preamble();

    const auto &set_accumulated_time_overload = static_cast<void (*)(snippets::op::PerfCountEnd*)>(set_accumulated_time);
    h->mov(h->rax, reinterpret_cast<size_t>(set_accumulated_time_overload));
    h->mov(abi_param1, reinterpret_cast<size_t>(m_end_node.get()));

    spill.rsp_align();
    h->call(h->rax);
    spill.rsp_restore();

    spill.postamble();
}

}   // namespace intel_cpu
}   // namespace ov
#endif // SNIPPETS_DEBUG_CAPS
