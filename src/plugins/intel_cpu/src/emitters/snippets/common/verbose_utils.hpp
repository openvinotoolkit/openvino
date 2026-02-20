// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef SNIPPETS_DEBUG_CAPS

#    pragma once

#    include <cstddef>
#    include <cstdlib>
#    include <memory>
#    include <sstream>
#    include <string>
#    include <typeinfo>

#    include "openvino/util/common_util.hpp"

#    ifndef _WIN32
#        include <cxxabi.h>
#    endif

namespace ov::intel_cpu::snippets_common {

/**
 * @brief Get demangled type name of a jit_emitter
 * @param emitter Pointer to the emitter object
 * @return Demangled type name string
 */
template <typename JitEmitter>
std::string get_emitter_type_name(const JitEmitter* emitter) {
    std::string name = typeid(*emitter).name();
#    ifndef _WIN32
    int status = 0;
    std::unique_ptr<char, void (*)(void*)> demangled_name(abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status),
                                                          std::free);
    if (status == 0 && demangled_name) {
        name = demangled_name.get();
    }
#    endif
    return name;
}

/**
 * @brief Format memory emitter info
 * @param emitter Pointer to memory emitter with src_prc, dst_prc, count, and compiled_byte_offset
 * @return Formatted string with memory emitter information
 */
template <typename MemoryEmitter>
std::string format_memory_emitter_info(const MemoryEmitter* emitter) {
    std::stringstream ss;
    ss << " src_precision:" << emitter->src_prc << " dst_precision:" << emitter->dst_prc
       << " load/store_element_number:" << emitter->count << " byte_offset:" << emitter->compiled_byte_offset;
    return ss.str();
}

/**
 * @brief Initialize info string for load memory emitter
 * @param emitter Pointer to load memory emitter
 * @return Formatted info string
 */
template <typename LoadMemoryEmitter>
std::string init_info_jit_load_memory_emitter(const LoadMemoryEmitter* emitter) {
    std::stringstream ss;
    std::string memory_emitter_info = format_memory_emitter_info(emitter);
    ss << "Emitter_type_name:jit_load_memory_emitter" << memory_emitter_info;
    return ss.str();
}

/**
 * @brief Initialize info string for load broadcast emitter
 * @param emitter Pointer to load broadcast emitter
 * @return Formatted info string
 */
template <typename LoadBroadcastEmitter>
std::string init_info_jit_load_broadcast_emitter(const LoadBroadcastEmitter* emitter) {
    std::stringstream ss;
    std::string memory_emitter_info = format_memory_emitter_info(emitter);
    ss << "Emitter_type_name:jit_load_broadcast_emitter" << memory_emitter_info;
    return ss.str();
}

/**
 * @brief Initialize info string for store memory emitter
 * @param emitter Pointer to store memory emitter
 * @return Formatted info string
 */
template <typename StoreMemoryEmitter>
std::string init_info_jit_store_memory_emitter(const StoreMemoryEmitter* emitter) {
    std::stringstream ss;
    std::string memory_emitter_info = format_memory_emitter_info(emitter);
    ss << "Emitter_type_name:jit_store_memory_emitter" << memory_emitter_info;
    return ss.str();
}

/**
 * @brief Base class for emitter info structure
 * Provides common initialization and access methods
 */
struct jit_emitter_info_base {
    jit_emitter_info_base() = default;
    jit_emitter_info_base(const jit_emitter_info_base& rhs) = default;
    jit_emitter_info_base& operator=(const jit_emitter_info_base& rhs) = default;
    virtual ~jit_emitter_info_base() = default;

    virtual void init(const void* emitter) = 0;

    [[nodiscard]] const char* c_str() const {
        return str_.c_str();
    }

    [[nodiscard]] bool is_initialized() const {
        return is_initialized_;
    }

protected:
    std::string str_;
    bool is_initialized_ = false;
};

}  // namespace ov::intel_cpu::snippets_common

#endif  // SNIPPETS_DEBUG_CAPS
