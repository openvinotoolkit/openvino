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

const std::vector<uint32_t> jit_emitter::store_gpr_regs = {
    // Parameter/result registers
    0, 1, 2, 3, 4, 5, 6, 7,
    // r8: Indirect result location register
    // r9...r15: Temporary registers
    9, 10, 11, 12, 13, 14, 15,
    16, 17, 18,
    // r19...r28: Callee-saved registers
    29, 30
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

std::set<std::vector<element::Type>> jit_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {};
}

size_t jit_emitter::get_aux_gprs_count() const {
    return 0;
}

size_t jit_emitter::get_max_vecs_count() const {
    return 32;
}

int32_t jit_emitter::get_vec_length() const {
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
        OPENVINO_THROW("Failed to allocate required number of vector registers");
    }

    if (pool_aux_gpr_idxs.size() < get_aux_gprs_count()) {
        OPENVINO_THROW("Failed to allocate required number of gpr registers");
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
    const int size = static_cast<int>(preserved_gpr_idxs.size());
    for (int i = (size - 1); i >= 0; --i) {
        h->ldr(Xbyak_aarch64::XReg(preserved_gpr_idxs[i]), post_ptr(h->sp, 16));
    }
    preserved_gpr_idxs.clear();

    aux_vec_idxs.clear();
    aux_gpr_idxs.clear();
}

void jit_emitter::store_context(const std::unordered_set<size_t>& ignore_registers) const {
    // 1. General-purpose Registers
    // 1.1. store pair registers
    const auto store_gpr_regs_size = store_gpr_regs.size();
    const auto last = store_gpr_regs_size % 2;
    for (size_t i = 0; i < (store_gpr_regs_size - last); i += 2) {
        h->stp(Xbyak_aarch64::XReg(store_gpr_regs[i]),
               Xbyak_aarch64::XReg(store_gpr_regs[i + 1]),
               pre_ptr(h->sp, -get_gpr_length() * 2));
    }

    // 1.1. store the remaining register
    if (last != 0) {
        h->str(Xbyak_aarch64::XReg(store_gpr_regs[store_gpr_regs_size - 1]),
               pre_ptr(h->sp, -get_gpr_length() * 2));
    }

    // 2. SIMD and Floating-Point registers
    // 2.1. store pair registers
    int prev_reg_idx = -1;
    size_t ignore_registers_count = 0;
    for (size_t reg_idx = 0; reg_idx < get_asimd_vectors_count(); reg_idx++) {
        if (ignore_registers.find(reg_idx) != ignore_registers.end()) {
            ignore_registers_count++;
            continue;
        }

        if (prev_reg_idx == -1) {
            prev_reg_idx = static_cast<int>(reg_idx);
            continue;
        }

        h->stp(Xbyak_aarch64::QReg(prev_reg_idx),
               Xbyak_aarch64::QReg(reg_idx),
               pre_ptr(h->sp, -get_vec_length() * 2));
        prev_reg_idx = -1;
    }
    OPENVINO_ASSERT(ignore_registers_count == ignore_registers.size(),
                    "ignored registers size is not equal actual ignored registers count");

    // 2.1. store the remaining register
    if (prev_reg_idx != -1) {
        h->str(Xbyak_aarch64::QReg(prev_reg_idx),
               pre_ptr(h->sp, -get_vec_length()));
    }
}

void jit_emitter::restore_context(const std::unordered_set<size_t>& ignore_registers) const {
    // 1. SIMD and Floating-Point registers
    // 1.1. restore the remaining register
    const auto v_last = (get_asimd_vectors_count() - ignore_registers.size()) % 2;
    if (v_last != 0) {
        const auto reg_idx = get_asimd_vectors_count() - 1;
        h->ldr(Xbyak_aarch64::QReg(reg_idx),
               post_ptr(h->sp, get_vec_length()));
    }

    // 2.2. restore pair registers
    size_t ignore_registers_count = 0;
    int prev_reg_idx = -1;
    for (size_t i = v_last; i < get_asimd_vectors_count(); i++) {
        const auto reg_idx = get_asimd_vectors_count() - 1 - i;
        if (ignore_registers.find(reg_idx) != ignore_registers.end()) {
            ignore_registers_count++;
            continue;
        }

        if (prev_reg_idx == -1) {
            prev_reg_idx = static_cast<int>(reg_idx);
            continue;
        }

        h->ldp(Xbyak_aarch64::QReg(reg_idx),
               Xbyak_aarch64::QReg(prev_reg_idx),
               post_ptr(h->sp, get_vec_length() * 2));
        prev_reg_idx = -1;
    }

    OPENVINO_ASSERT(ignore_registers_count == ignore_registers.size(),
                    "ignored registers size is not equal actual ignored registers count");

    // 2. General-purpose Registers
    // 2.1. restore the remaining register
    const auto save_gpr_regs_size = store_gpr_regs.size();
    const auto last = save_gpr_regs_size % 2;
    if (last != 0) {
        h->ldr(Xbyak_aarch64::XReg(store_gpr_regs[save_gpr_regs_size - 1]),
               post_ptr(h->sp, get_gpr_length() * 2));
    }

    // 2.2. restore pair registers
    for (size_t i = last; i < save_gpr_regs_size; i += 2) {
        h->ldp(Xbyak_aarch64::XReg(store_gpr_regs[save_gpr_regs_size - 1 - (i + 1)]),
               Xbyak_aarch64::XReg(store_gpr_regs[save_gpr_regs_size - 1 - i]),
               post_ptr(h->sp, get_gpr_length() * 2));
    }
}

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
