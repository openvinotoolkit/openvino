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

const std::vector<uint32_t> jit_emitter::save_gpr_regs = {
    9, 10
};

const std::vector<uint32_t> jit_emitter::save_v_regs = {
    10, 11, 16, 17, 18, 19, 20, 21, 22, 23
};

void jit_emitter::emit_code(const std::vector<size_t> &in_idxs,
                            const std::vector<size_t> &out_idxs,
                            const std::vector<size_t> &pool_vec_idxs,
                            const std::vector<size_t> &pool_gpr_idxs) const {
    emitter_preamble(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs);

    emit_impl(in_idxs, out_idxs);

    emitter_postamble();
}

void jit_emitter::emit_data() const {
    h->align(64);
    h->L(*l_table.get());

    // Assumption: entries can be inserted with dd, so they should be 4 bytes.
    assert(sizeof(table_entry_val_t) == 4);

    // Run through the map and insert values stored there
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        const auto &te = (*it).second; // get map entry for a given key
        const auto len = te.bcast ? get_vec_length() : sizeof(table_entry_val_t);
        for (size_t d = 0; d < len; d += sizeof(table_entry_val_t))
            h->dd(te.val);
    }
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
    register_table_entries();

    // Now that we registered the entries, we set the offsets.  No
    // entries should be registered after this point.  This allows to
    // expect the same order when injecting the table entries in
    // prepare_table.
    size_t off = 0;
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        auto &te = (*it).second;
        te.off = off;
        off += te.bcast ? get_vec_length() : sizeof(table_entry_val_t);
    }
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
        preserved_gpr_idxs.push_back(static_cast<uint32_t>(idx));
    }

    if (!entry_map_.empty()) {
        // last aux_gpr_idx is for p_table, we can use aux_gpr_idxs from idx 0 for other purpose
        p_table = Xbyak_aarch64::XReg(aux_gpr_idxs[aux_gpr_idxs.size() - 1]);
        aux_gpr_idxs.erase(aux_gpr_idxs.end() - 1);
    }

    for (size_t i = 0; i < preserved_gpr_idxs.size(); ++i) {
        h->str(Xbyak_aarch64::XReg(preserved_gpr_idxs[i]), pre_ptr(h->sp, -16));
    }

    if (!entry_map_.empty()) {
        load_table_addr();
    }
}

void jit_emitter::emitter_postamble() const {
    for (size_t i = 0; i < preserved_gpr_idxs.size(); ++i) {
        h->ldr(Xbyak_aarch64::XReg(preserved_gpr_idxs[i]), post_ptr(h->sp, 16));
    }
    preserved_gpr_idxs.clear();

    aux_vec_idxs.clear();
    aux_gpr_idxs.clear();
}

void jit_emitter::store_context() const {
    // X29: The register x29 represents the base pointer (also known as the frame pointer or FP)
    // X30: In A64 systems, the return address is stored in register x30 (also known as LR)
    h->stp(h->x29, h->x30, pre_ptr(h->sp, -16));

    // General-purpose Registers
    const auto save_gpr_regs_size = save_gpr_regs.size();
    const int32_t xreg_len = 8;
    for (size_t i = 0; i < save_gpr_regs_size; i += 2) {
        h->stp(Xbyak_aarch64::XReg(save_gpr_regs[i]),
            Xbyak_aarch64::XReg(save_gpr_regs[i + 1]),
            pre_ptr(h->sp, -xreg_len * 2));
    }

    // SIMD and Floating-Point registers
    const auto save_v_regs_size = save_v_regs.size();
    const int32_t qreg_len = 16;
    for (size_t i = 0; i < save_v_regs_size; i += 2) {
        h->stp(Xbyak_aarch64::QReg(save_v_regs[i]),
                Xbyak_aarch64::QReg(save_v_regs[i + 1]),
                pre_ptr(h->sp, -qreg_len * 2));
    }
}

void jit_emitter::restore_context() const {
    // SIMD and Floating-Point registers
    const auto save_v_regs_size = save_v_regs.size();
    const int32_t qreg_len = 16;
    for (size_t i = 0; i < save_v_regs_size; i += 2) {
        h->ldp(Xbyak_aarch64::QReg(save_v_regs[save_v_regs_size - 1 - (i + 1)]),
                Xbyak_aarch64::QReg(save_v_regs[save_v_regs_size - 1 - i]),
                post_ptr(h->sp, qreg_len * 2));
    }

    // General-purpose Registers
    const auto save_gpr_regs_size = save_gpr_regs.size();
    const int32_t xreg_len = 8;
    for (size_t i = 0; i < save_gpr_regs_size; i += 2) {
        h->ldp(Xbyak_aarch64::XReg(save_gpr_regs[save_gpr_regs_size - 1 - (i + 1)]),
                Xbyak_aarch64::XReg(save_gpr_regs[save_gpr_regs_size - 1 - i]),
                post_ptr(h->sp, xreg_len * 2));
    }

    h->ldp(h->x29, h->x30, post_ptr(h->sp, 16));
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
