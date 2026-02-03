// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#ifdef SNIPPETS_DEBUG_CAPS

#    include "emitters/snippets/common/verbose_utils.hpp"
#    include "jit_brgemm_copy_b_emitter.hpp"
#    include "jit_brgemm_emitter.hpp"
#    include "jit_kernel_emitter.hpp"
#    include "jit_memory_emitters.hpp"
#    include "jit_segfault_detector_emitter.hpp"
#    include "kernel_executors/brgemm.hpp"
#    include "kernel_executors/brgemm_amx.hpp"
#    include "openvino/util/common_util.hpp"
#    include "verbose.hpp"

namespace ov::intel_cpu {

std::string init_info_jit_brgemm_emitter(const jit_brgemm_emitter* emitter) {
    std::stringstream ss;
    ss << "Emitter_type_name:jit_brgemm_emitter";
    if (const auto& common = std::dynamic_pointer_cast<x64::BrgemmKernelExecutor>(emitter->m_kernel_executor)) {
        ss << common->to_string();
    }
    if (const auto& amx = std::dynamic_pointer_cast<x64::BrgemmAMXKernelExecutor>(emitter->m_kernel_executor)) {
        ss << amx->to_string();
    }
    ss << " m_memory_offset:" << ov::util::vector_to_string(emitter->m_memory_offsets)
       << " m_buffer_ids:" << ov::util::vector_to_string(emitter->m_buffer_ids);

    return ss.str();
}

std::string init_info_jit_brgemm_copy_b_emitter(const jit_brgemm_copy_b_emitter* emitter) {
    std::stringstream ss;
    ss << "Emitter_type_name:jit_brgemm_copy_b_emitter" << emitter->m_kernel_executor->to_string()
       << " m_memory_offset:" << ov::util::vector_to_string(emitter->m_memory_offsets)
       << " m_buffer_ids:" << ov::util::vector_to_string(emitter->m_buffer_ids);

    return ss.str();
}

std::string init_info_jit_kernel_static_emitter(const jit_kernel_static_emitter* emitter) {
    std::stringstream ss;
    ss << "Emitter_type_name:jit_kernel_static_emitter"
       << " jcp.exec_domain:" << ov::util::vector_to_string(emitter->jcp.exec_domain)
       << " master_shape:" << ov::util::vector_to_string(emitter->master_shape) << " num_inputs:" << emitter->num_inputs
       << " num_outputs:" << emitter->num_outputs << " num_unique_buffers:" << emitter->num_unique_buffers
       << " data_ptr_regs_idx:" << ov::util::vector_to_string(emitter->data_ptr_regs_idx);
    for (size_t i = 0; i < emitter->data_offsets.size(); ++i) {
        ss << " data_offsets for " << i << " is:" << ov::util::vector_to_string(emitter->data_offsets[i]);
    }
    return ss.str();
}

std::string init_info_jit_kernel_dynamic_emitter(const jit_kernel_dynamic_emitter* emitter) {
    std::stringstream ss;
    ss << "Emitter_type_name:jit_kernel_dynamic_emitter" << " num_inputs:" << emitter->num_inputs
       << " num_outputs:" << emitter->num_outputs << " num_unique_buffers:" << emitter->num_unique_buffers
       << " data_ptr_regs_idx:" << ov::util::vector_to_string(emitter->data_ptr_regs_idx);
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
    // traget emitter info
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

void jit_emitter_info_t::init(const void* emitter) {
    if (is_initialized_) {
        return;
    }
    const auto* e = static_cast<const jit_emitter*>(emitter);
    if (const auto* e_type = dynamic_cast<const jit_load_memory_emitter*>(e)) {
        str_ = snippets_common::init_info_jit_load_memory_emitter(e_type);
    } else if (const auto* e_type = dynamic_cast<const jit_load_broadcast_emitter*>(e)) {
        str_ = snippets_common::init_info_jit_load_broadcast_emitter(e_type);
    } else if (const auto* e_type = dynamic_cast<const jit_store_memory_emitter*>(e)) {
        str_ = snippets_common::init_info_jit_store_memory_emitter(e_type);
    } else if (const auto* e_type = dynamic_cast<const jit_brgemm_emitter*>(e)) {
        str_ = init_info_jit_brgemm_emitter(e_type);
    } else if (const auto* e_type = dynamic_cast<const jit_brgemm_copy_b_emitter*>(e)) {
        str_ = init_info_jit_brgemm_copy_b_emitter(e_type);
    } else if (const auto* e_type = dynamic_cast<const jit_kernel_static_emitter*>(e)) {
        str_ = init_info_jit_kernel_static_emitter(e_type);
    } else if (const auto* e_type = dynamic_cast<const jit_kernel_dynamic_emitter*>(e)) {
        str_ = init_info_jit_kernel_dynamic_emitter(e_type);
    } else if (const auto* e_type = dynamic_cast<const jit_uni_segfault_detector_emitter*>(e)) {
        str_ = init_info_jit_uni_segfault_detector_emitter(e_type);
    } else {
        str_ = init_info_jit_emitter_general(e);
    }
    is_initialized_ = true;
}

}  // namespace ov::intel_cpu

#endif
