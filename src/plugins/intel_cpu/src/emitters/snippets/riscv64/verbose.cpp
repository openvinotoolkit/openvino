// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <cstdlib>
#include <sstream>
#include <string>
#ifdef SNIPPETS_DEBUG_CAPS

#    include "emitters/snippets/common/verbose_utils.hpp"
#    include "jit_kernel_emitter.hpp"
#    include "jit_memory_emitters.hpp"
#    include "jit_segfault_detector_emitter.hpp"
#    include "verbose.hpp"

namespace ov::intel_cpu::riscv64 {

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

std::string init_info_jit_kernel_emitter(const jit_kernel_emitter* emitter) {
    std::stringstream ss;
    ss << " jcp.exec_domain:" << snippets_common::vector_to_string(emitter->jcp.exec_domain)
       << " num_inputs:" << emitter->num_inputs << " num_outputs:" << emitter->num_outputs
       << " num_unique_buffers:" << emitter->num_unique_buffers
       << " data_ptr_regs_idx:" << snippets_common::vector_to_string(emitter->data_ptr_regs_idx);
    return ss.str();
}

std::string init_info_jit_kernel_static_emitter(const jit_kernel_static_emitter* emitter) {
    std::stringstream ss;
    ss << "Emitter_type_name:jit_kernel_static_emitter" << init_info_jit_kernel_emitter(emitter)
       << " master_shape:" << snippets_common::vector_to_string(emitter->master_shape);
    for (size_t i = 0; i < emitter->data_offsets.size(); ++i) {
        ss << " data_offsets for " << i << " is:" << snippets_common::vector_to_string(emitter->data_offsets[i]);
    }
    return ss.str();
}

std::string init_info_jit_kernel_dynamic_emitter(const jit_kernel_dynamic_emitter* emitter) {
    std::stringstream ss;
    ss << "Emitter_type_name:jit_kernel_dynamic_emitter" << init_info_jit_kernel_emitter(emitter);
    return ss.str();
}

std::string init_info_jit_uni_segfault_detector_emitter(const jit_uni_segfault_detector_emitter* emitter) {
    std::stringstream ss;
    ss << "Node_name:" << emitter->m_target_node_name << " use_load_emitter:" << emitter->is_target_use_load_emitter
       << " use_store_emitter:" << emitter->is_target_use_store_emitter;
    if (emitter->is_target_use_load_emitter || emitter->is_target_use_store_emitter) {
        ss << " start_address:" << emitter->start_address << " current_address:" << emitter->current_address
           << " iteration:" << emitter->iteration << " ";
    }
    if (const auto* target_e = emitter->get_target_emitter()) {
        ss << target_e->info();
    }
    return ss.str();
}

static std::string init_info_jit_emitter_general(const jit_emitter* emitter) {
    std::stringstream ss;
    ss << "Emitter_type_name:" << snippets_common::get_emitter_type_name(emitter);
    return ss.str();
}

void jit_emitter_info_t::init(const jit_emitter* emitter) {
    if (is_initialized_) {
        return;
    }
    if (const auto* e_type = dynamic_cast<const jit_load_memory_emitter*>(emitter)) {
        str_ = init_info_jit_load_memory_emitter(e_type);
    } else if (const auto* e_type = dynamic_cast<const jit_load_broadcast_emitter*>(emitter)) {
        str_ = init_info_jit_load_broadcast_emitter(e_type);
    } else if (const auto* e_type = dynamic_cast<const jit_store_memory_emitter*>(emitter)) {
        str_ = init_info_jit_store_memory_emitter(e_type);
    } else if (const auto* e_type = dynamic_cast<const jit_kernel_static_emitter*>(emitter)) {
        str_ = init_info_jit_kernel_static_emitter(e_type);
    } else if (const auto* e_type = dynamic_cast<const jit_kernel_dynamic_emitter*>(emitter)) {
        str_ = init_info_jit_kernel_dynamic_emitter(e_type);
    } else if (const auto* e_type = dynamic_cast<const jit_uni_segfault_detector_emitter*>(emitter)) {
        str_ = init_info_jit_uni_segfault_detector_emitter(e_type);
    } else {
        str_ = init_info_jit_emitter_general(emitter);
    }
    is_initialized_ = true;
}

}  // namespace ov::intel_cpu::riscv64

#endif
