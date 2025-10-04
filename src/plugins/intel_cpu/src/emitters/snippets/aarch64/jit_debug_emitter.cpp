// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#ifdef SNIPPETS_DEBUG_CAPS

#    include <vector>

#    include "jit_debug_emitter.hpp"
#    include "utils/general_utils.h"

using namespace dnnl::impl::cpu;
using namespace dnnl::impl;

namespace ov::intel_cpu::aarch64 {

void jit_debug_emitter::emit_data() const {
    m_target_emitter->emit_data();
}

void jit_debug_emitter::emit_code_impl(const std::vector<size_t>& in_idxs,
                                       const std::vector<size_t>& out_idxs,
                                       const std::vector<size_t>& pool_vec_idxs,
                                       const std::vector<size_t>& pool_gpr_idxs) const {
    if (any_of(m_decorator_emit_loc, EmissionLocation::preamble, EmissionLocation::both)) {
        m_decorator_emitter->emit_code(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs);
    }

    m_target_emitter->emit_code(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs);

    if (any_of(m_decorator_emit_loc, EmissionLocation::postamble, EmissionLocation::both)) {
        m_decorator_emitter->emit_code(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs);
    }
}

void jit_debug_emitter::emit_impl(const std::vector<size_t>& /*in_idxs*/,
                                  const std::vector<size_t>& /*out_idxs*/) const {
    // No-op: debug emitter delegates via emit_code_impl
}

size_t jit_debug_emitter::get_inputs_count() const {
    return m_target_emitter->get_inputs_count();
}

size_t jit_debug_emitter::get_aux_vecs_count() const {
    return m_target_emitter->get_aux_vecs_count();
}

size_t jit_debug_emitter::get_aux_gprs_count() const {
    return m_target_emitter->get_aux_gprs_count();
}

void jit_debug_emitter::prepare_table() {
    m_target_emitter->prepare_table();
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

void jit_debug_emitter::validate_arguments(const std::vector<size_t>& /*arg0*/,
                                           const std::vector<size_t>& /*arg1*/) const {
}

}  // namespace ov::intel_cpu::aarch64

#endif
