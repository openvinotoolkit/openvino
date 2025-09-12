// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#ifdef SNIPPETS_DEBUG_CAPS

#    include "jit_segfault_detector_emitter.hpp"
#    include "verbose.hpp"

#    ifndef _WIN32
#        include <cxxabi.h>
#    endif

namespace ov::intel_cpu::aarch64 {

using ov::intel_cpu::aarch64::jit_emitter;  // base type lives in aarch64

std::string init_info_jit_uni_segfault_detector_emitter(
    const jit_uni_segfault_detector_emitter* emitter) {
    std::stringstream ss;
    ss << "Node_name:" << emitter->m_target_node_name << " use_load_emitter:" << emitter->is_target_use_load_emitter
       << " use_store_emitter:" << emitter->is_target_use_store_emitter << ' ' << get_segfault_tracking_info();
    if (const auto* target_e = emitter->get_target_emitter()) {
        ss << ' ' << target_e->info();
    }
    return ss.str();
}

std::string get_emitter_type_name(const jit_emitter* emitter) {
    std::string name = typeid(*emitter).name();
#    ifndef _WIN32
    int status = 0;
    std::unique_ptr<char, void (*)(void*)> demangled_name(abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status),
                                                          std::free);
    if (demangled_name) name = demangled_name.get();
#    endif
    return name;
}

static std::string init_info_jit_emitter_general(const jit_emitter* emitter) {
    std::stringstream ss;
    ss << "Emitter_type_name:" << get_emitter_type_name(emitter);
    return ss.str();
}

}  // namespace ov::intel_cpu::aarch64

namespace ov::intel_cpu {

using aarch64::jit_emitter;
using aarch64::jit_uni_segfault_detector_emitter;

void jit_emitter_info_t::init(const jit_emitter* emitter) {
    if (is_initialized_) return;
    if (const auto* e_type = dynamic_cast<const jit_uni_segfault_detector_emitter*>(emitter)) {
        str_ = aarch64::init_info_jit_uni_segfault_detector_emitter(e_type);
    } else {
        str_ = aarch64::init_info_jit_emitter_general(emitter);
    }
    is_initialized_ = true;
}

}  // namespace ov::intel_cpu

#endif
