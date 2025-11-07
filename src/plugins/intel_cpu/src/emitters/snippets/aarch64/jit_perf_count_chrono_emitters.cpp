// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_perf_count_chrono_emitters.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_reg.h>

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "jit_binary_call_emitter.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/perf_count.hpp"

#ifdef SNIPPETS_DEBUG_CAPS

using namespace dnnl::impl;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::aarch64;
using namespace Xbyak_aarch64;

namespace ov::intel_cpu::aarch64 {

jit_perf_count_chrono_start_emitter::jit_perf_count_chrono_start_emitter(
    dnnl::impl::cpu::aarch64::jit_generator* host,
    dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
    const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_binary_call_emitter(host, host_isa, expr->get_live_regs()) {
    m_start_node = ov::as_type_ptr<ov::snippets::op::PerfCountBegin>(expr->get_node());
    OPENVINO_ASSERT(m_start_node, "PerfCountBegin node is null");
}

void jit_perf_count_chrono_start_emitter::set_start_time(ov::snippets::op::PerfCountBegin* start_node) {
    start_node->set_start_time();
}

void jit_perf_count_chrono_start_emitter::emit_impl([[maybe_unused]] const std::vector<size_t>& in_idxs,
                                                    [[maybe_unused]] const std::vector<size_t>& out_idxs) const {
    // Initialize call-specific regs according to ARM64 AAPCS and live regs
    init_binary_call_regs(/*num_binary_args*/ 1, /*used_gpr_idxs*/ {});
    const XReg& func_reg = get_call_address_reg();
    const auto& fn_ptr =
        reinterpret_cast<uint64_t>(static_cast<void (*)(ov::snippets::op::PerfCountBegin*)>(set_start_time));

    // Conservatively preserve full JIT context across external call
    store_context({});
    h->mov(func_reg, fn_ptr);
    h->mov(XReg(0), reinterpret_cast<uint64_t>(m_start_node.get()));
    h->blr(func_reg);
    restore_context({});
}

jit_perf_count_chrono_end_emitter::jit_perf_count_chrono_end_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                                                     const ov::snippets::lowered::ExpressionPtr& expr)
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
    const XReg& func_reg = get_call_address_reg();
    const auto& fn_ptr =
        reinterpret_cast<uint64_t>(static_cast<void (*)(ov::snippets::op::PerfCountEnd*)>(set_accumulated_time));

    store_context({});
    h->mov(func_reg, fn_ptr);
    h->mov(XReg(0), reinterpret_cast<uint64_t>(m_end_node.get()));
    h->blr(func_reg);
    restore_context({});
}

}  // namespace ov::intel_cpu::aarch64

#endif  // SNIPPETS_DEBUG_CAPS
