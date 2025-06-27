// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "openvino/runtime/threading/thread_local.hpp"
#ifdef SNIPPETS_DEBUG_CAPS

#    include <utility>

#    include "jit_segfault_detector_emitter.hpp"

using namespace dnnl::impl::utils;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::aarch64;
using namespace Xbyak_aarch64;

namespace ov::intel_cpu::aarch64 {

std::shared_ptr<ThreadLocal<jit_uni_segfault_detector_emitter*>> g_custom_segfault_handler =
    std::make_shared<ThreadLocal<jit_uni_segfault_detector_emitter*>>();

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
    std::unordered_set<size_t> exclude = {};
    store_context(exclude);

    const auto& set_local_handler_overload =
        static_cast<void (*)(jit_uni_segfault_detector_emitter*)>(set_local_handler);
    XReg func_reg(8);
    h->mov(func_reg, reinterpret_cast<size_t>(set_local_handler_overload));
    h->mov(XReg(0), reinterpret_cast<size_t>(this));
    h->blr(func_reg);

    restore_context(exclude);
}

void jit_uni_segfault_detector_emitter::set_local_handler(jit_uni_segfault_detector_emitter* emitter_address) {
    g_custom_segfault_handler->local() = emitter_address;
}

void jit_uni_segfault_detector_emitter::memory_track(size_t gpr_idx) const {
    Xbyak_aarch64::Label set_current, done;

    // ——— 1) Save X15 (and dummy XZR) in a 16-byte aligned way ———
    //    SP -= 16; STP X15, XZR, [SP]
    h->stp(XReg(15), XReg(31), pre_ptr(h->sp, -16));

    // ——— 2) if (start_address != 0) jump to set_current ———
    h->mov(XReg(15), reinterpret_cast<uint64_t>(&start_address));
    h->ldr(XReg(0), ptr(XReg(15)));
    h->cbnz(XReg(0), set_current);

    // ——— 3) start_address = [gpr_idx] ———
    h->str(XReg(gpr_idx), ptr(XReg(15)));

    // ——— 4) current_address = [gpr_idx], then jump to done ———
    h->mov(XReg(15), reinterpret_cast<uint64_t>(&current_address));
    h->str(XReg(gpr_idx), ptr(XReg(15)));
    h->b(done);

    // ——— 5) label: set_current ———
    h->L(set_current);
    h->mov(XReg(15), reinterpret_cast<uint64_t>(&current_address));
    h->str(XReg(gpr_idx), ptr(XReg(15)));

    // ——— 6) label: done — bump iteration ———
    h->L(done);
    h->mov(XReg(15), reinterpret_cast<uint64_t>(&iteration));
    h->ldr(XReg(0), ptr(XReg(15)));
    h->add(XReg(0), XReg(0), 1);
    h->str(XReg(0), ptr(XReg(15)));

    // ——— 7) Restore X15 (and XZR), SP += 16 ———
    h->ldp(XReg(15), XReg(31), post_ptr(h->sp, 16));
}

}  // namespace ov::intel_cpu::aarch64

#endif
