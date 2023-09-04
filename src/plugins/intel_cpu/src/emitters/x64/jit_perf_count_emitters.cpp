// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_emitter.hpp"
#include "jit_perf_count_emitters.hpp"
#include <cpu/x64/jit_generator.hpp>

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;
using namespace Xbyak::util;

namespace ov {
namespace intel_cpu {

static void get_current_time(std::chrono::high_resolution_clock::time_point* current_time) {
    *current_time = std::chrono::high_resolution_clock::now();
}

static void get_accumulated_time(std::chrono::high_resolution_clock::time_point* start_time, uint64_t* accumulation, uint32_t* num) {
    auto current_time = std::chrono::high_resolution_clock::now();
    *accumulation += std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - *start_time).count();
    (*num)++;
}

jit_perf_count_start_emitter::jit_perf_count_start_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
                                            const std::shared_ptr<ov::Node>& n) : jit_emitter(host, host_isa, n) {
    auto start_op = ov::as_type_ptr<snippets::op::PerfCountBegin>(n);
    m_current_time = &(start_op->start_time_stamp);
}

size_t jit_perf_count_start_emitter::get_inputs_num() const {
    return 0;
}

void jit_perf_count_start_emitter::emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    internal_call_preamble();

    h->push(h->rax);
    h->push(abi_param1);

    h->mov(abi_param1, reinterpret_cast<size_t>(m_current_time));
    h->mov(h->rax, reinterpret_cast<size_t>(&get_current_time));

    internal_call_rsp_align();
    h->call(h->rax);
    internal_call_rsp_restore();

    h->pop(abi_param1);
    h->pop(h->rax);

    internal_call_postamble();
}

///////////////////jit_perf_count_end_emitter////////////////////////////////////
jit_perf_count_end_emitter::jit_perf_count_end_emitter(dnnl::impl::cpu::x64::jit_generator *host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
    const std::shared_ptr<ov::Node>& n) : jit_emitter(host, host_isa, n) {
        auto end_op = ov::as_type_ptr<snippets::op::PerfCountEnd>(n);
        m_accumulation = &(end_op->accumulation);
        m_iteration = &(end_op->iteration);
        m_start = &(end_op->perf_count_start.start_time_stamp);
}

size_t jit_perf_count_end_emitter::get_inputs_num() const {
    return 0;
}

void jit_perf_count_end_emitter::emit_impl(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs) const {
    internal_call_preamble();

    h->push(h->rax);
    h->push(abi_param1);
    h->push(abi_param2);
    h->push(abi_param3);

    h->mov(abi_param1, reinterpret_cast<size_t>(m_start));
    h->mov(abi_param2, reinterpret_cast<size_t>(m_accumulation));
    h->mov(abi_param3, reinterpret_cast<size_t>(m_iteration));
    h->mov(h->rax, reinterpret_cast<size_t>(&get_accumulated_time));

    internal_call_rsp_align();
    h->call(h->rax);
    internal_call_rsp_restore();

    h->pop(abi_param3);
    h->pop(abi_param2);
    h->pop(abi_param1);
    h->pop(h->rax);

    internal_call_postamble();
}

}   // namespace intel_cpu
}   // namespace ov
