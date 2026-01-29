// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef SNIPPETS_DEBUG_CAPS

#    include "jit_segfault_detector_emitter.hpp"

#    include <cpu/aarch64/cpu_isa_traits.hpp>
#    include <cpu/aarch64/jit_generator.hpp>
#    include <cstddef>
#    include <cstdint>
#    include <cstdlib>
#    include <sstream>
#    include <string>
#    include <unordered_set>
#    include <utility>

#    include "emitters/plugin/aarch64/jit_emitter.hpp"
#    include "emitters/snippets/common/jit_segfault_detector_emitter_base.hpp"
#    include "verbose.hpp"
#    include "xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_adr.h"
#    include "xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_gen.h"
#    include "xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_label.h"
#    include "xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_reg.h"

using namespace dnnl::impl::cpu::aarch64;
using namespace Xbyak_aarch64;

namespace ov::intel_cpu::aarch64 {

jit_uni_segfault_detector_emitter::jit_uni_segfault_detector_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                                                     dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                                                     jit_emitter* target_emitter,
                                                                     bool is_load,
                                                                     bool is_store,
                                                                     std::string target_node_name)
    : base_t(host, host_isa, target_emitter, is_load, is_store, std::move(target_node_name)) {}

size_t jit_uni_segfault_detector_emitter::get_inputs_count() const {
    return 1;
}

const jit_emitter* jit_uni_segfault_detector_emitter::get_target_emitter() const {
    return base_t::get_target_emitter();
}

std::string jit_uni_segfault_detector_emitter::info() const {
    std::stringstream ss;
    ss << "Node_name:" << m_target_node_name << " use_load_emitter:" << is_target_use_load_emitter
       << " use_store_emitter:" << is_target_use_store_emitter;
    if (is_target_use_load_emitter || is_target_use_store_emitter) {
        ss << " start_address:" << start_address << " current_address:" << current_address << " iteration:" << iteration
           << " ";
    }
    if (const auto* target_e = get_target_emitter()) {
        jit_emitter_info_t info;
        info.init(target_e);
        ss << info.c_str();
    }
    return ss.str();
}

bool jit_uni_segfault_detector_emitter::save_before_memory_track() const {
    // aarch64 tracks and saves lazily inside memory_track() on the first iteration
    return false;
}

void jit_uni_segfault_detector_emitter::save_target_emitter() const {
    if (ov::intel_cpu::g_custom_segfault_handler<jit_uni_segfault_detector_emitter>->local() == this) {
        return;
    }
    std::unordered_set<size_t> ignore_vec_regs;
    ignore_vec_regs.reserve(get_max_vecs_count());
    for (size_t i = 0; i < get_max_vecs_count(); ++i) {
        ignore_vec_regs.insert(i);
    }
    store_context(ignore_vec_regs);

    const auto& set_local_handler_overload =
        static_cast<void (*)(jit_uni_segfault_detector_emitter*)>(set_local_handler);
    const auto fn_ptr = reinterpret_cast<uint64_t>(set_local_handler_overload);
    h->mov(h->X_TMP_0, fn_ptr);
    h->mov(XReg(0), reinterpret_cast<uint64_t>(this));
    h->blr(h->X_TMP_0);
    restore_context(ignore_vec_regs);
}

void jit_uni_segfault_detector_emitter::set_local_handler(jit_uni_segfault_detector_emitter* emitter_address) {
    base_t::set_local_handler_impl(ov::intel_cpu::g_custom_segfault_handler<jit_uni_segfault_detector_emitter>,
                                   emitter_address);
}

void jit_uni_segfault_detector_emitter::memory_track(size_t gpr_idx_for_mem_address) const {
    XReg addr_reg(16);
    XReg val_reg(17);
    XReg mem_addr_reg(15);

    h->sub(h->sp, h->sp, 48);
    h->stp(mem_addr_reg, addr_reg, ptr(h->sp, 0));
    h->str(val_reg, ptr(h->sp, 16));
    h->mrs(val_reg, 3, 3, 4, 2, 0);
    h->str(val_reg, ptr(h->sp, 24));

    if (gpr_idx_for_mem_address == static_cast<size_t>(mem_addr_reg.getIdx())) {
        h->ldr(mem_addr_reg, ptr(h->sp, 0));
    } else if (gpr_idx_for_mem_address == static_cast<size_t>(addr_reg.getIdx())) {
        h->ldr(mem_addr_reg, ptr(h->sp, 8));
    } else if (gpr_idx_for_mem_address == static_cast<size_t>(val_reg.getIdx())) {
        h->ldr(mem_addr_reg, ptr(h->sp, 16));
    } else {
        h->mov(mem_addr_reg, XReg(static_cast<int>(gpr_idx_for_mem_address)));
    }

    Xbyak_aarch64::Label label_fast_path;
    Xbyak_aarch64::Label label_done;

    h->mov(addr_reg, reinterpret_cast<uint64_t>(&iteration));
    h->ldr(val_reg, ptr(addr_reg));
    h->cmp(val_reg, 0);
    h->b(NE, label_fast_path);

    save_target_emitter();

    h->mov(addr_reg, reinterpret_cast<uint64_t>(&start_address));
    h->str(mem_addr_reg, ptr(addr_reg));
    h->mov(addr_reg, reinterpret_cast<uint64_t>(&current_address));
    h->str(mem_addr_reg, ptr(addr_reg));
    h->mov(addr_reg, reinterpret_cast<uint64_t>(&iteration));
    h->mov(val_reg, 1);
    h->str(val_reg, ptr(addr_reg));
    h->b(label_done);

    h->L(label_fast_path);
    h->add_imm(val_reg, val_reg, 1, mem_addr_reg);
    h->str(val_reg, ptr(addr_reg));
    h->mov(addr_reg, reinterpret_cast<uint64_t>(&current_address));
    h->str(mem_addr_reg, ptr(addr_reg));
    h->L(label_done);

    h->ldr(val_reg, ptr(h->sp, 24));
    h->msr(3, 3, 4, 2, 0, val_reg);
    h->ldp(mem_addr_reg, addr_reg, ptr(h->sp, 0));
    h->ldr(val_reg, ptr(h->sp, 16));
    h->add(h->sp, h->sp, 48);
}

}  // namespace ov::intel_cpu::aarch64

#endif
