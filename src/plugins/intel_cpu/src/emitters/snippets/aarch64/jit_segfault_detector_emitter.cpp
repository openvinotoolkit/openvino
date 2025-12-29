// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef SNIPPETS_DEBUG_CAPS

#    include "jit_segfault_detector_emitter.hpp"

#    include <cpu/aarch64/cpu_isa_traits.hpp>
#    include <cpu/aarch64/jit_generator.hpp>
#    include <cstddef>
#    include <cstdint>
#    include <cstdlib>
#    include <memory>
#    include <sstream>
#    include <string>
#    include <typeinfo>
#    include <utility>
#    include <vector>

#    include "emitters/plugin/aarch64/jit_emitter.hpp"
#    include "emitters/snippets/aarch64/jit_memory_emitters.hpp"
#    include "xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_adr.h"
#    include "xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_gen.h"
#    include "xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_label.h"
#    include "xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_reg.h"

#    ifndef _WIN32
#        include <cxxabi.h>
#    endif

using namespace dnnl::impl::cpu::aarch64;
using namespace Xbyak_aarch64;

namespace ov::intel_cpu::aarch64 {

const std::shared_ptr<ThreadLocal<jit_uni_segfault_detector_emitter*>> g_custom_segfault_handler =
    std::make_shared<ThreadLocal<jit_uni_segfault_detector_emitter*>>();

static std::string get_emitter_type_name(const jit_emitter* emitter) {
    std::string name = typeid(*emitter).name();
#    ifndef _WIN32
    int status = 0;
    std::unique_ptr<char, void (*)(void*)> demangled_name(abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status),
                                                          std::free);
    name = demangled_name.get();
#    endif
    return name;
}

std::string init_info_jit_memory_emitter(const jit_memory_emitter* emitter) {
    std::stringstream ss;
    ss << " src_precision:" << emitter->src_prc << " dst_precision:" << emitter->dst_prc
       << " load/store_element_number:" << emitter->count << " byte_offset:" << emitter->compiled_byte_offset;
    return ss.str();
}

static std::string init_info_jit_load_memory_emitter(const jit_load_memory_emitter* emitter) {
    std::stringstream ss;
    std::string memory_emitter_info = init_info_jit_memory_emitter(emitter);
    ss << "Emitter_type_name:jit_load_memory_emitter" << memory_emitter_info;
    return ss.str();
}

static std::string init_info_jit_load_broadcast_emitter(const jit_load_broadcast_emitter* emitter) {
    std::stringstream ss;
    std::string memory_emitter_info = init_info_jit_memory_emitter(emitter);
    ss << "Emitter_type_name:jit_load_broadcast_emitter" << memory_emitter_info;
    return ss.str();
}

static std::string init_info_jit_store_memory_emitter(const jit_store_memory_emitter* emitter) {
    std::stringstream ss;
    std::string memory_emitter_info = init_info_jit_memory_emitter(emitter);
    ss << "Emitter_type_name:jit_store_memory_emitter" << memory_emitter_info;
    return ss.str();
}

static std::string init_info_jit_emitter_general(const jit_emitter* emitter) {
    std::stringstream ss;
    ss << "Emitter_type_name:" << get_emitter_type_name(emitter);
    return ss.str();
}

static std::string init_info_jit_target_emitter(const jit_emitter* emitter) {
    if (const auto* e_type = dynamic_cast<const jit_load_memory_emitter*>(emitter)) {
        return init_info_jit_load_memory_emitter(e_type);
    }
    if (const auto* e_type = dynamic_cast<const jit_load_broadcast_emitter*>(emitter)) {
        return init_info_jit_load_broadcast_emitter(e_type);
    }
    if (const auto* e_type = dynamic_cast<const jit_store_memory_emitter*>(emitter)) {
        return init_info_jit_store_memory_emitter(e_type);
    }
    return init_info_jit_emitter_general(emitter);
}

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

std::string jit_uni_segfault_detector_emitter::info() const {
    std::stringstream ss;
    ss << "Node_name:" << m_target_node_name << " use_load_emitter:" << is_target_use_load_emitter
       << " use_store_emitter:" << is_target_use_store_emitter;
    if (is_target_use_load_emitter || is_target_use_store_emitter) {
        ss << " start_address:" << start_address << " current_address:" << current_address << " iteration:" << iteration
           << " ";
    }
    if (const auto* target_e = get_target_emitter()) {
        ss << init_info_jit_target_emitter(target_e);
    }
    return ss.str();
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
    store_context({});

    const auto& set_local_handler_overload =
        static_cast<void (*)(jit_uni_segfault_detector_emitter*)>(set_local_handler);
    const auto fn_ptr = reinterpret_cast<uint64_t>(set_local_handler_overload);
    h->mov(h->X_TMP_0, fn_ptr);
    h->mov(XReg(0), reinterpret_cast<uint64_t>(this));
    h->blr(h->X_TMP_0);

    restore_context({});
}

void jit_uni_segfault_detector_emitter::set_local_handler(jit_uni_segfault_detector_emitter* emitter_address) {
    g_custom_segfault_handler->local() = emitter_address;
}

void jit_uni_segfault_detector_emitter::memory_track(size_t gpr_idx_for_mem_address) const {
    XReg addr_reg(16);
    XReg val_reg(17);
    XReg mem_addr_reg(15);

    h->sub(h->sp, h->sp, 32);
    h->stp(mem_addr_reg, addr_reg, ptr(h->sp, 0));
    h->str(val_reg, ptr(h->sp, 16));

    if (gpr_idx_for_mem_address == static_cast<size_t>(mem_addr_reg.getIdx())) {
        h->ldr(mem_addr_reg, ptr(h->sp, 0));
    } else if (gpr_idx_for_mem_address == static_cast<size_t>(addr_reg.getIdx())) {
        h->ldr(mem_addr_reg, ptr(h->sp, 8));
    } else if (gpr_idx_for_mem_address == static_cast<size_t>(val_reg.getIdx())) {
        h->ldr(mem_addr_reg, ptr(h->sp, 16));
    } else {
        h->mov(mem_addr_reg, XReg(static_cast<int>(gpr_idx_for_mem_address)));
    }

    Xbyak_aarch64::Label label_set_address_current;
    Xbyak_aarch64::Label label_set_address_end;

    h->mov(addr_reg, reinterpret_cast<uint64_t>(&start_address));
    h->ldr(val_reg, ptr(addr_reg));
    h->cmp(val_reg, 0);
    h->b(NE, label_set_address_current);
    h->str(mem_addr_reg, ptr(addr_reg));
    h->mov(addr_reg, reinterpret_cast<uint64_t>(&current_address));
    h->str(mem_addr_reg, ptr(addr_reg));
    h->b(label_set_address_end);

    h->L(label_set_address_current);
    h->mov(addr_reg, reinterpret_cast<uint64_t>(&current_address));
    h->str(mem_addr_reg, ptr(addr_reg));
    h->L(label_set_address_end);

    h->mov(addr_reg, reinterpret_cast<uint64_t>(&iteration));
    h->ldr(val_reg, ptr(addr_reg));
    h->add_imm(val_reg, val_reg, 1, mem_addr_reg);
    h->str(val_reg, ptr(addr_reg));

    h->ldp(mem_addr_reg, addr_reg, ptr(h->sp, 0));
    h->ldr(val_reg, ptr(h->sp, 16));
    h->add(h->sp, h->sp, 32);
}

}  // namespace ov::intel_cpu::aarch64

#endif
