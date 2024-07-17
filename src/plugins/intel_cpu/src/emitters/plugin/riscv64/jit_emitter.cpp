// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_emitter.hpp"
#include <vector>
#include "utils/general_utils.h"
#include "emitters/utils.hpp"

namespace ov {
namespace intel_cpu {
namespace riscv64 {

const std::vector<size_t> jit_emitter::store_gpr_regs = {};

static const std::vector<size_t> vec_regs = {};

jit_emitter::jit_emitter(ov::intel_cpu::riscv64::jit_generator* host,
                         ov::element::Type exec_prc,
                         emitter_in_out_map in_out_type) :
                         Emitter(),
                         h(host),
                         exec_prc_(exec_prc),
                         in_out_type_(in_out_type),
                         p_table(0),
                         l_table(new Xbyak_riscv::Label()) {
}

jit_emitter::jit_emitter(ov::intel_cpu::riscv64::jit_generator* host,
                         const std::shared_ptr<ov::Node>& n,
                         ov::element::Type exec_prc,
                         emitter_in_out_map in_out_type)
                         : Emitter(), h(host), exec_prc_(exec_prc),
    in_out_type_(in_out_type), p_table(0), l_table (new Xbyak_riscv::Label()) {
}

void jit_emitter::emit_code(const std::vector<size_t> &in_idxs,
                            const std::vector<size_t> &out_idxs,
                            const std::vector<size_t> &pool_vec_idxs,
                            const std::vector<size_t> &pool_gpr_idxs) const {
    emitter_preamble(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs);

    emit_impl(in_idxs, out_idxs);

    emitter_postamble();
}

void jit_emitter::emit_data() const {
     h->L(*l_table.get());

     // Assumption: entries can be inserted with dd, so they should be 4 bytes.
     assert(sizeof(table_entry_val_t) == 4);

     // Run through the map and insert values stored there
     for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
         const auto &te = (*it).second; // get map entry for a given key
         const auto len = te.bcast ? get_vec_length() : sizeof(table_entry_val_t);
         for (size_t d = 0; d < len; d += sizeof(table_entry_val_t))
             h->append4B(te.val);
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
        OPENVINO_THROW("Failed to allocate required number of gpr registers. Pool size: " +
            std::to_string(pool_aux_gpr_idxs.size()) + ", required size: " + std::to_string(get_aux_gprs_count()));
    }

    const bool is_vec_input = (in_out_type_ == emitter_in_out_map::vec_to_vec) ||
                              (in_out_type_ == emitter_in_out_map::vec_to_gpr);
    const bool is_vec_output = (in_out_type_ == emitter_in_out_map::vec_to_vec) ||
                               (in_out_type_ == emitter_in_out_map::gpr_to_vec);

    // vector registers
    for (auto idx : pool_aux_vec_idxs) {
        aux_vec_idxs.push_back(static_cast<uint32_t>(idx));
    }

    for (size_t idx = 0; idx < get_max_vecs_count(); idx++) {
        if (aux_vec_idxs.size() >= get_aux_vecs_count()) break;

        if (is_vec_input) {
            if (std::find(in_idxs.begin(), in_idxs.end(), idx) != in_idxs.end()) continue;
        }
        if (is_vec_output) {
            if (std::find(out_idxs.begin(), out_idxs.end(), idx) != out_idxs.end()) continue;
        }

        if (std::find(in_idxs.begin(), in_idxs.end(), idx) != in_idxs.end()) continue;
        if (std::find(out_idxs.begin(), out_idxs.end(), idx) != out_idxs.end()) continue;

        if (std::find(aux_vec_idxs.begin(), aux_vec_idxs.end(), idx) != aux_vec_idxs.end()) continue;
        if (std::find(preserved_vec_idxs.begin(), preserved_vec_idxs.end(), idx) != preserved_vec_idxs.end()) continue;

        aux_vec_idxs.push_back(idx);
        preserved_vec_idxs.push_back(idx);
    }
    if (aux_vec_idxs.size() < get_aux_vecs_count())
        OV_CPU_JIT_EMITTER_THROW("Failed to allocate required number of vector registers");

    // gpr registers
    for (auto idx : pool_aux_gpr_idxs) {
        aux_gpr_idxs.push_back(idx);
    }

    const uint32_t end_gpr_idx = Xbyak_riscv::x31.getIdx();
    for (size_t gpr_idx = 0; gpr_idx <= end_gpr_idx; ++gpr_idx) {
        size_t _idx = end_gpr_idx - gpr_idx; // we allocate from the end

        if (aux_gpr_idxs.size() >= get_aux_gprs_count()) break;

        if (!is_vec_input) {
            if (std::find(in_idxs.begin(), in_idxs.end(), _idx) != in_idxs.end()) continue;
        }
        if (!is_vec_output) {
            if (std::find(out_idxs.begin(), out_idxs.end(), _idx) != out_idxs.end()) continue;
        }

        if (std::find(aux_gpr_idxs.begin(), aux_gpr_idxs.end(), _idx) != aux_gpr_idxs.end()) continue;
        if (std::find(preserved_gpr_idxs.begin(), preserved_gpr_idxs.end(), _idx) != preserved_gpr_idxs.end()) continue;

        aux_gpr_idxs.push_back(_idx);
        preserved_gpr_idxs.push_back(_idx);
    }
    if (aux_gpr_idxs.size() < get_aux_gprs_count())
        OV_CPU_JIT_EMITTER_THROW("Failed to allocate required number of general-purpose registers");

    if (!entry_map_.empty()) {
        // last aux_gpr_idx is for p_table, we can use aux_gpr_idxs from idx 0 for other purpose
        p_table = Xbyak_riscv::Reg(aux_gpr_idxs[aux_gpr_idxs.size() - 1]);
        aux_gpr_idxs.erase(aux_gpr_idxs.end() - 1);
    }

    store_context(preserved_gpr_idxs, preserved_vec_idxs);

    if (!entry_map_.empty()) {
        load_table_addr();
    }
}

void jit_emitter::emitter_postamble() const {
    restore_context(preserved_gpr_idxs, preserved_vec_idxs);

    preserved_vec_idxs.clear();
    preserved_gpr_idxs.clear();

    aux_vec_idxs.clear();
    aux_gpr_idxs.clear();
}

void jit_emitter::store_context(const std::unordered_set<size_t>& ignore_registers) const {
    store_context(store_gpr_regs, vec_regs, ignore_registers);
}

void jit_emitter::store_context(
        const std::vector<size_t>& gpr_regs,
        const std::vector<size_t>& vec_regs,
        const std::unordered_set<size_t>& ignore_vec_regs) const {
}

void jit_emitter::restore_context(const std::unordered_set<size_t>& ignore_vec_regs) const {
    restore_context(store_gpr_regs, vec_regs, ignore_vec_regs);
}

void jit_emitter::restore_context(
        const std::vector<size_t>& gpr_regs,
        const std::vector<size_t>& vec_regs,
        const std::unordered_set<size_t>& ignore_vec_regs) const {
}

void jit_emitter::load(const VReg& rd, const float imm, const Reg& aux) const {
    h->li(aux, dnnl::impl::float2int(imm));
    FReg aux_f(0);
    h->fmv_w_x(aux_f, aux);

    h->vxor_vv(rd, rd, rd);
    h->vfadd_vf(rd, rd, aux_f);
}

void jit_emitter::load_table_val(const std::string& key, const Reg& aux, const VReg& target, const size_t key_off_val_shift) const {
    const int32_t off = table_off(key, key_off_val_shift);
    h->addi(aux, p_table, off);

    // TODO: workaround: broadcast load
    h->flw(f0, aux);
    h->vxor_vv(target, target, target);
    h->vfadd_vf(target, target, f0);
}

}   // namespace riscv64
}   // namespace intel_cpu
}   // namespace ov
