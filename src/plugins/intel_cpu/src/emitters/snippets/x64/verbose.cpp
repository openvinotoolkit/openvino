// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef SNIPPETS_DEBUG_CAPS

#include "verbose.hpp"
#include "jit_segfault_detector_emitter.hpp"
#include "jit_memory_emitters.hpp"
#include "jit_brgemm_emitter.hpp"
#include "jit_brgemm_copy_b_emitter.hpp"
#include "jit_kernel_emitter.hpp"
#include "jit_snippets_emitters.hpp"

#ifndef _WIN32
#include <cxxabi.h>
#endif

namespace ov {
namespace intel_cpu {

template <typename T>
std::string join(const T& v, const std::string& sep = ", ") {
    std::ostringstream ss;
    size_t count = 0;
    for (const auto& x : v) {
        if (count++ > 0) {
            ss << sep;
        }
        ss << x;
    }
    return ss.str();
}

template <typename T>
std::string vector_to_string(const T& v) {
    std::ostringstream os;
    os << "[ " << ov::util::join(v) << " ]";
    return os.str();
}

std::string get_emitter_type_name(const jit_emitter* emitter) {
    std::string name = typeid(*emitter).name();
#ifndef _WIN32
    int status;
    std::unique_ptr<char, void (*)(void*)> demangled_name(
            abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status),
            std::free);
    name = demangled_name.get();
#endif
    return name;
}

std::string init_info_jit_memory_emitter(const jit_memory_emitter *emitter) {
    std::stringstream ss;
    ss << " src_precision:" << emitter->src_prc
       << " dst_precision:" << emitter->dst_prc
       << " load/store_element_number:" << emitter->count
       << " byte_offset:" << emitter->compiled_byte_offset;
    return ss.str();
}

static std::string init_info_jit_load_memory_emitter(const jit_load_memory_emitter *emitter) {
    std::stringstream ss;
    std::string memory_emitter_info = init_info_jit_memory_emitter(emitter);
    ss << "Emitter_type_name:jit_load_memory_emitter"
       << memory_emitter_info;
    return ss.str();
}

static std::string init_info_jit_load_broadcast_emitter(const jit_load_broadcast_emitter *emitter) {
    std::stringstream ss;
    std::string memory_emitter_info = init_info_jit_memory_emitter(emitter);
    ss << "Emitter_type_name:jit_load_broadcast_emitter"
       << memory_emitter_info;
    return ss.str();
}

static std::string init_info_jit_store_memory_emitter(const jit_store_memory_emitter *emitter) {
    std::stringstream ss;
    std::string memory_emitter_info = init_info_jit_memory_emitter(emitter);
    ss << "Emitter_type_name:jit_store_memory_emitter"
       << memory_emitter_info;
    return ss.str();
}

std::string init_info_jit_brgemm_emitter(const jit_brgemm_emitter *emitter) {
    std::stringstream ss;
    ss << "Emitter_type_name:jit_brgemm_emitter"
       <<  emitter->m_kernel_executor->to_string()
       << " m_memory_offset:" << vector_to_string(emitter->m_memory_offsets)
       << " m_buffer_ids:" << vector_to_string(emitter->m_buffer_ids);

    return ss.str();
}

std::string init_info_jit_brgemm_copy_b_emitter(const jit_brgemm_copy_b_emitter *emitter) {
    std::stringstream ss;
    ss << "Emitter_type_name:jit_brgemm_copy_b_emitter"
       <<  emitter->m_kernel_executor->to_string()
       << " m_memory_offset:" << vector_to_string(emitter->m_memory_offsets)
       << " m_buffer_ids:" << vector_to_string(emitter->m_buffer_ids);

    return ss.str();
}

std::string init_info_jit_kernel_static_emitter(const jit_kernel_static_emitter* emitter) {
    std::stringstream ss;
    ss << "Emitter_type_name:jit_kernel_static_emitter"
       << " jcp.exec_domain:" << vector_to_string(emitter->jcp.exec_domain)
       << " gp_regs_pool:"<< vector_to_string(emitter->gp_regs_pool)
       << " master_shape:" << vector_to_string(emitter->master_shape)
       << " num_inputs:" << emitter->num_inputs
       << " num_outputs:" << emitter->num_outputs
       << " num_unique_buffers:" << emitter->num_unique_buffers
       << " data_ptr_regs_idx:" << vector_to_string(emitter->data_ptr_regs_idx)
       << " vec_regs_pool:" << vector_to_string(emitter->vec_regs_pool)
       << " reg_indexes_idx:" << emitter->reg_indexes_idx
       << " reg_runtime_params_idx:" << emitter->reg_runtime_params_idx;
    for (size_t i = 0; i < emitter->data_offsets.size(); ++i)
        ss << " data_offsets for " << i << " is:" << vector_to_string(emitter->data_offsets[i]);
    return ss.str();
}

std::string init_info_jit_kernel_dynamic_emitter(const jit_kernel_dynamic_emitter* emitter) {
    std::stringstream ss;
    ss << "Emitter_type_name:jit_kernel_dynamic_emitter"
       << " gp_regs_pool:"<< vector_to_string(emitter->gp_regs_pool)
       << " num_inputs:" << emitter->num_inputs
       << " num_outputs:" << emitter->num_outputs
       << " num_unique_buffers:" << emitter->num_unique_buffers
       << " data_ptr_regs_idx:" << vector_to_string(emitter->data_ptr_regs_idx)
       << " vec_regs_pool:" << vector_to_string(emitter->vec_regs_pool)
       << " reg_runtime_params_idx:" << emitter->reg_runtime_params_idx;
    return ss.str();
}

std::string init_info_jit_uni_segfault_detector_emitter(const jit_uni_segfault_detector_emitter *emitter) {
    std::stringstream ss;
    ss << "Node_name:" << emitter->m_target_node_name
       << " use_load_emitter:"<< emitter->is_target_use_load_emitter
       << " use_store_emitter:"<< emitter->is_target_use_store_emitter;
    if (emitter->is_target_use_load_emitter || emitter->is_target_use_store_emitter) {
        ss << " start_address:" << emitter->start_address
           << " current_address:" << emitter->current_address
           << " iteration:" << emitter->iteration << " ";
    }
    // traget emitter info
    if (auto target_e = emitter->get_target_emitter()) {
        ss << target_e->info();
    }
    return ss.str();
}

static std::string init_info_jit_emitter_general(const jit_emitter *emitter) {
    std::stringstream ss;
    ss << "Emitter_type_name:" << get_emitter_type_name(emitter);
    return ss.str();
}

void jit_emitter_info_t::init(const jit_emitter *emitter) {
    if (is_initialized_) return;
    if (auto e_type = dynamic_cast<const jit_load_memory_emitter*>(emitter)) {
        str_ = init_info_jit_load_memory_emitter(e_type);
    } else if (auto e_type = dynamic_cast<const jit_load_broadcast_emitter*>(emitter)) {
        str_ = init_info_jit_load_broadcast_emitter(e_type);
    } else if (auto e_type = dynamic_cast<const jit_store_memory_emitter*>(emitter)) {
        str_ = init_info_jit_store_memory_emitter(e_type);
    } else if (auto e_type = dynamic_cast<const jit_brgemm_emitter*>(emitter)) {
        str_ = init_info_jit_brgemm_emitter(e_type);
    } else if (auto e_type = dynamic_cast<const jit_brgemm_copy_b_emitter*>(emitter)) {
        str_ = init_info_jit_brgemm_copy_b_emitter(e_type);
    } else if (auto e_type = dynamic_cast<const jit_kernel_static_emitter*>(emitter)) {
        str_ = init_info_jit_kernel_static_emitter(e_type);
    } else if (auto e_type = dynamic_cast<const jit_kernel_dynamic_emitter*>(emitter)) {
        str_ = init_info_jit_kernel_dynamic_emitter(e_type);
    } else if (auto e_type = dynamic_cast<const jit_uni_segfault_detector_emitter*>(emitter)) {
        str_ = init_info_jit_uni_segfault_detector_emitter(e_type);
    } else {
        str_ = init_info_jit_emitter_general(emitter);
    }
    is_initialized_ = true;
}

}   // namespace intel_cpu
}   // namespace ov

#endif