// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef SNIPPETS_DEBUG_CAPS

#    include "verbose_utils.hpp"

#    include <sstream>
#    include <string>

namespace ov::intel_cpu::snippets_common {

/**
 * @brief Format memory emitter info
 *
 * This function formats common memory emitter information that is shared across
 * all architectures (x64, aarch64, riscv64). The memory emitter struct is expected
 * to have: src_prc, dst_prc, count, and compiled_byte_offset fields.
 */
template <typename MemoryEmitter>
std::string format_memory_emitter_info(const MemoryEmitter* emitter) {
    std::stringstream ss;
    ss << " src_precision:" << emitter->src_prc << " dst_precision:" << emitter->dst_prc
       << " load/store_element_number:" << emitter->count << " byte_offset:" << emitter->compiled_byte_offset;
    return ss.str();
}

/**
 * @brief Format kernel emitter common info
 *
 * This function formats common kernel emitter information that is shared across
 * all architectures.
 */
template <typename KernelEmitter>
std::string format_kernel_emitter_info(const KernelEmitter* emitter) {
    std::stringstream ss;
    ss << " jcp.exec_domain:" << vector_to_string(emitter->jcp.exec_domain) << " num_inputs:" << emitter->num_inputs
       << " num_outputs:" << emitter->num_outputs << " num_unique_buffers:" << emitter->num_unique_buffers
       << " data_ptr_regs_idx:" << vector_to_string(emitter->data_ptr_regs_idx);
    return ss.str();
}

/**
 * @brief Format segfault detector emitter info
 */
template <typename SegfaultDetectorEmitter>
std::string format_segfault_detector_info(const SegfaultDetectorEmitter* emitter) {
    std::stringstream ss;
    ss << "Node_name:" << emitter->m_target_node_name << " use_load_emitter:" << emitter->is_target_use_load_emitter
       << " use_store_emitter:" << emitter->is_target_use_store_emitter;
    if (emitter->is_target_use_load_emitter || emitter->is_target_use_store_emitter) {
        ss << " start_address:" << emitter->start_address << " current_address:" << emitter->current_address
           << " iteration:" << emitter->iteration << " ";
    }
    return ss.str();
}

/**
 * @brief Format general emitter type info
 */
template <typename Emitter>
std::string format_emitter_general(const Emitter* emitter) {
    std::stringstream ss;
    ss << "Emitter_type_name:" << get_type_name(emitter);
    return ss.str();
}

// Explicit template instantiations are handled in architecture-specific files

}  // namespace ov::intel_cpu::snippets_common

#endif  // SNIPPETS_DEBUG_CAPS
