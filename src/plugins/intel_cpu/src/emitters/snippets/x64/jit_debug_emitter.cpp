// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef SNIPPETS_DEBUG_CAPS

#    include "jit_debug_emitter.hpp"

#    include <vector>

#    include "utils/general_utils.h"

using namespace dnnl::impl::cpu;
using namespace dnnl::impl;
using namespace Xbyak;

namespace ov::intel_cpu {

size_t jit_debug_emitter::get_inputs_num() const {
    return m_target_emitter->get_inputs_num();
}

size_t jit_debug_emitter::aux_vecs_count() const {
    return m_target_emitter->aux_vecs_count();
}

size_t jit_debug_emitter::aux_gprs_count() const {
    return m_target_emitter->aux_gprs_count();
}

void jit_debug_emitter::emitter_preamble(const std::vector<size_t>& in_idxs,
                                         const std::vector<size_t>& out_idxs,
                                         const std::vector<size_t>& pool_vec_idxs,
                                         const std::vector<size_t>& pool_gpr_idxs) const {
    m_target_emitter->emitter_preamble(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs);
}

void jit_debug_emitter::emitter_postamble() const {
    m_target_emitter->emitter_postamble();
}

void jit_debug_emitter::validate_arguments(const std::vector<size_t>& arg0, const std::vector<size_t>& arg1) const {
    m_target_emitter->validate_arguments(arg0, arg1);
}

void jit_debug_emitter::emit_data() const {
    m_target_emitter->emit_data();
}

void jit_debug_emitter::prepare_table() {
    m_target_emitter->prepare_table();
}

void jit_debug_emitter::register_table_entries() {
    m_target_emitter->register_table_entries();
}

void jit_debug_emitter::emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const {
    m_target_emitter->emit_impl(in_idxs, out_idxs);
}

void jit_debug_emitter::emit_code_impl(const std::vector<size_t>& in_idxs,
                                       const std::vector<size_t>& out_idxs,
                                       const std::vector<size_t>& pool_vec_idxs,
                                       const std::vector<size_t>& pool_gpr_idxs) const {
    if (m_decorator_emit_loc == EmissionLocation::preamble || m_decorator_emit_loc == EmissionLocation::both) {
        m_decorator_emitter->emit_code(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs);
    }

    m_target_emitter->emit_code(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs);

    if (m_decorator_emit_loc == EmissionLocation::postamble || m_decorator_emit_loc == EmissionLocation::both) {
        m_decorator_emitter->emit_code(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs);
    }
}

}  // namespace ov::intel_cpu

#endif