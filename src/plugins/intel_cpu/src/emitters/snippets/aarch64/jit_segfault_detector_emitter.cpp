// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <cstdint>
#include <string>

#ifdef SNIPPETS_DEBUG_CAPS

#    include <utility>

// Include our header first to pull in jit_generator and dependencies
#    include "jit_segfault_detector_emitter.hpp"

// Then include Xbyak AArch64 building blocks
#    include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_adr.h>
#    include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_gen.h>
#    include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_label.h>
#    include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_reg.h>

#    include <cpu/aarch64/cpu_isa_traits.hpp>
#    include <cpu/aarch64/jit_generator.hpp>

#    include "emitters/plugin/aarch64/jit_emitter.hpp"

using namespace dnnl::impl::utils;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::aarch64;
using namespace Xbyak_aarch64;

namespace ov::intel_cpu::aarch64 {

const std::shared_ptr<ThreadLocal<jit_uni_segfault_detector_emitter*>> g_custom_segfault_handler =
    std::make_shared<ThreadLocal<jit_uni_segfault_detector_emitter*>>();

// Keep tracked info in process-global storage to avoid lifetime
// issues and per-thread TLS address resolution in JIT code.
static size_t s_start_address = 0;
static size_t s_current_address = 0;
static size_t s_iteration = 0;

jit_uni_segfault_detector_emitter::jit_uni_segfault_detector_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                                                     jit_emitter* target_emitter,
                                                                     bool is_load,
                                                                     bool is_store,
                                                                     std::string target_node_name)
    : jit_emitter(host, host_isa),
      m_target_emitter(target_emitter),
      is_target_use_load_emitter(is_load),
      is_target_use_store_emitter(is_store),
      m_target_node_name(std::move(target_node_name)) {}

size_t jit_uni_segfault_detector_emitter::get_inputs_count() const {
    return 1;
}

const jit_emitter* jit_uni_segfault_detector_emitter::get_target_emitter() const {
    return m_target_emitter;
}

void jit_uni_segfault_detector_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs,
                                                  const std::vector<size_t>& out_vec_idxs) const {
    save_target_emitter();
    if (is_target_use_load_emitter) {
        memory_track(in_vec_idxs[0]);
    } else if (is_target_use_store_emitter) {
        memory_track(out_vec_idxs[0]);
    }
}

void jit_uni_segfault_detector_emitter::save_target_emitter() const {
    // Disabled external call on AArch64 for stability; thread-local
    // handler is not required for normal runs and memory_track is enough.
}

void jit_uni_segfault_detector_emitter::set_local_handler(jit_uni_segfault_detector_emitter* emitter_address) {
    g_custom_segfault_handler->local() = emitter_address;
}

void jit_uni_segfault_detector_emitter::memory_track(size_t gpr_idx_for_mem_address) const {
    Label label_set_address_current;
    Label label_set_address_end;

    // if (start_address == 0) start_address = <mem_addr>; current_address = <mem_addr>; else current_address =
    // <mem_addr>;
    XReg r_addr(10);
    XReg r_val(11);
    XReg r_mem(static_cast<int>(gpr_idx_for_mem_address));

    // Preserve scratch regs used in this helper (x10, x11)
    h->sub(h->sp, h->sp, 16);
    h->stp(r_addr, r_val, ptr(h->sp));

    h->mov(r_addr, reinterpret_cast<size_t>(&s_start_address));
    h->ldr(r_val, ptr(r_addr));
    h->cmp(r_val, 0);
    h->b(NE, label_set_address_current);
    h->str(r_mem, ptr(r_addr));
    h->mov(r_addr, reinterpret_cast<size_t>(&s_current_address));
    h->str(r_mem, ptr(r_addr));
    h->b(label_set_address_end);
    h->L(label_set_address_current);
    {
        h->mov(r_addr, reinterpret_cast<size_t>(&s_current_address));
        h->str(r_mem, ptr(r_addr));
    }
    h->L(label_set_address_end);
    // iteration++, 1 means first access
    h->mov(r_addr, reinterpret_cast<size_t>(&s_iteration));
    h->ldr(r_val, ptr(r_addr));
    h->add(r_val, r_val, 0x01);
    h->str(r_val, ptr(r_addr));

    // Restore scratch regs
    h->ldp(r_addr, r_val, ptr(h->sp));
    h->add(h->sp, h->sp, 16);
}

}  // namespace ov::intel_cpu::aarch64

#endif
