// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_emitter.hpp"
#include <vector>
#include "utils/general_utils.h"

using namespace dnnl::impl::cpu;
using namespace dnnl::impl;

namespace ov {
namespace intel_cpu {
namespace aarch64 {

void jit_emitter::emit_code(const std::vector<size_t> &in_idxs,
                            const std::vector<size_t> &out_idxs,
                            const std::vector<size_t> &pool_vec_idxs,
                            const std::vector<size_t> &pool_gpr_idxs) const {
    emitter_preamble(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs);

    emit_impl(in_idxs, out_idxs);

    emitter_postamble();
}

std::set<std::vector<element::Type>> jit_emitter::get_supported_precisions(const std::shared_ptr<ngraph::Node>& node) {
    return {};
}

size_t jit_emitter::get_aux_gprs_count() const {
    return 0;
}

size_t jit_emitter::get_max_vecs_count() const {
    return 32;
}

size_t jit_emitter::get_vec_length() const {
    return 16;
}

size_t jit_emitter::get_aux_vecs_count() const {
    return 0;
}

void jit_emitter::prepare_table() {
}

void jit_emitter::emitter_preamble(const std::vector<size_t>& in_idxs,
                                   const std::vector<size_t>& out_idxs,
                                   const std::vector<size_t>& pool_aux_vec_idxs,
                                   const std::vector<size_t>& pool_aux_gpr_idxs) const {
    if (pool_aux_vec_idxs.size() < get_aux_vecs_count()) {
        IE_THROW() << "Failed to allocate required number of vector registers";
    }

    if (pool_aux_gpr_idxs.size() < get_aux_gprs_count()) {
        IE_THROW() << "Failed to allocate required number of gpr registers";
    }

    for (auto idx : pool_aux_vec_idxs) {
        aux_vec_idxs.push_back(static_cast<uint32_t>(idx));
    }

    for (auto idx : pool_aux_gpr_idxs) {
        aux_gpr_idxs.push_back(static_cast<uint32_t>(idx));
    }
}

void jit_emitter::emitter_postamble() const {
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
