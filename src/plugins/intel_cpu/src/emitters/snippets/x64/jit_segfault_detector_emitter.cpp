// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef SNIPPETS_DEBUG_CAPS

#include "jit_segfault_detector_emitter.hpp"
#include "emitters/plugin/x64/utils.hpp"

using namespace dnnl::impl::utils;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {

std::shared_ptr<ThreadLocal<jit_uni_segfault_detector_emitter*>> g_custom_segfault_handler =
    std::make_shared<ThreadLocal<jit_uni_segfault_detector_emitter*>>();

jit_uni_segfault_detector_emitter::jit_uni_segfault_detector_emitter(dnnl::impl::cpu::x64::jit_generator* host, dnnl::impl::cpu::x64::cpu_isa_t host_isa,
    jit_emitter* target_emitter, bool is_load, bool is_store, std::string target_node_name) :
    jit_emitter(host, host_isa),
    m_target_emitter(target_emitter),
    is_target_use_load_emitter(is_load),
    is_target_use_store_emitter(is_store),
    m_target_node_name(target_node_name) {
}

size_t jit_uni_segfault_detector_emitter::get_inputs_num() const { return 1; }

const jit_emitter* jit_uni_segfault_detector_emitter::get_target_emitter() const {
    return m_target_emitter;
}

void jit_uni_segfault_detector_emitter::emit_impl(const std::vector<size_t>& in_vec_idxs, const std::vector<size_t>& out_vec_idxs) const {
    save_target_emitter();
    if (is_target_use_load_emitter) {
        memory_track(in_vec_idxs[0]);
    } else if (is_target_use_store_emitter) {
        memory_track(out_vec_idxs[0]);
    }
}

void jit_uni_segfault_detector_emitter::save_target_emitter() const {
    // use internal call as "->local" shoule be the execution thread. Otherwise always compilation thread.
    EmitABIRegSpills spill(h);
    spill.preamble();

    const auto &set_local_handler_overload = static_cast<void (*)(jit_uni_segfault_detector_emitter*)>(set_local_handler);
    h->mov(h->rax, reinterpret_cast<size_t>(set_local_handler_overload));
    h->mov(abi_param1, reinterpret_cast<uint64_t>(this));

    spill.rsp_align();
    h->call(h->rax);
    spill.rsp_restore();

    spill.postamble();
}

void jit_uni_segfault_detector_emitter::set_local_handler(jit_uni_segfault_detector_emitter* emitter_address) {
    g_custom_segfault_handler->local() = emitter_address;
}

void jit_uni_segfault_detector_emitter::memory_track(size_t gpr_idx_for_mem_address) const {
    h->push(h->r15);
    Xbyak::Label label_set_address_current;
    Xbyak::Label label_set_address_end;
    h->mov(h->r15, reinterpret_cast<size_t>(&start_address));
    h->cmp(h->qword[h->r15], 0);
    h->jne(label_set_address_current);
    h->mov(h->qword[h->r15], Xbyak::Reg64(gpr_idx_for_mem_address));
    h->mov(h->r15, reinterpret_cast<size_t>(&current_address));
    h->mov(h->qword[h->r15], Xbyak::Reg64(gpr_idx_for_mem_address));
    h->jmp(label_set_address_end);
    h->L(label_set_address_current);
    {
        h->mov(h->r15, reinterpret_cast<size_t>(&current_address));
        h->mov(h->qword[h->r15], Xbyak::Reg64(gpr_idx_for_mem_address));
    }
    h->L(label_set_address_end);
    // iteration++, 1 means first access
    h->mov(h->r15, reinterpret_cast<size_t>(&iteration));
    h->add(h->qword[h->r15], 0x01);
    h->pop(h->r15);
}

}   // namespace intel_cpu
}   // namespace ov

#endif
