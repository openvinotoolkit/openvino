// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_emitter.hpp"

namespace ov {
namespace intel_cpu {
namespace riscv64 {

using namespace Xbyak_riscv;

jit_emitter::jit_emitter(ov::intel_cpu::riscv64::jit_generator* host, ov::intel_cpu::riscv64::cpu_isa_t host_isa,
                         ov::element::Type exec_prc, emitter_in_out_map in_out_type)
    : Emitter(), h(host), host_isa_(host_isa), exec_prc_(exec_prc), in_out_type_(in_out_type) {}

void jit_emitter::emit_code(const std::vector<size_t>& in_idxs,
                            const std::vector<size_t>& out_idxs,
                            const std::vector<size_t>& pool_vec_idxs,
                            const std::vector<size_t>& pool_gpr_idxs) const {
    emit_code(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs, {});
}

void jit_emitter::emit_code(const std::vector<size_t>& in_idxs,
                            const std::vector<size_t>& out_idxs,
                            const std::vector<size_t>& pool_vec_idxs,
                            const std::vector<size_t>& pool_gpr_idxs,
                            const std::vector<size_t>& pool_fp_gpr_idxs) const {
    emitter_preamble(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs, pool_fp_gpr_idxs);

    emit_impl(in_idxs, out_idxs);

    emitter_postamble();
}

size_t jit_emitter::aux_vecs_count() const {
    return 0;
}

size_t jit_emitter::aux_gprs_count() const {
    // We need one gpr to load table address
    return entry_map_.empty() ? 0 : 1;
}

size_t jit_emitter::aux_fp_gprs_count() const {
    return 0;
}

emitter_in_out_map jit_emitter::get_in_out_type() const {
    return in_out_type_;
}

std::set<std::vector<element::Type>> jit_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    return {};
}

size_t jit_emitter::get_gpr_length() const {
    return xlen;
}

size_t jit_emitter::get_fp_gpr_length() const {
    return flen;
}

size_t jit_emitter::get_vec_length() const {
    return vlen;
}

void jit_emitter::emitter_preamble(const std::vector<size_t>& in_idxs,
                                   const std::vector<size_t>& out_idxs,
                                   const std::vector<size_t>& pool_vec_idxs,
                                   const std::vector<size_t>& pool_gpr_idxs,
                                   const std::vector<size_t>& pool_fp_gpr_idxs) const {
    bool is_vec_input =
        (in_out_type_ == emitter_in_out_map::vec_to_vec) || (in_out_type_ == emitter_in_out_map::vec_to_gpr);
    bool is_vec_output =
        (in_out_type_ == emitter_in_out_map::vec_to_vec) || (in_out_type_ == emitter_in_out_map::gpr_to_vec);

    for (auto idx : pool_vec_idxs) {
        aux_vec_idxs.push_back(idx);
    }

    for (size_t idx = 0; idx < get_max_vecs_count(); idx++) {
        if (aux_vec_idxs.size() >= aux_vecs_count()) {
            break;
        }

        if (is_vec_input) {
            if (std::find(in_idxs.begin(), in_idxs.end(), idx) != in_idxs.end()) {
                continue;
            }
        }
        if (is_vec_output) {
            if (std::find(out_idxs.begin(), out_idxs.end(), idx) != out_idxs.end()) {
                continue;
            }
        }
        if (std::find(aux_vec_idxs.begin(), aux_vec_idxs.end(), idx) != aux_vec_idxs.end()) {
            continue;
        }
        if (std::find(preserved_vec_idxs.begin(), preserved_vec_idxs.end(), idx) != preserved_vec_idxs.end()) {
            continue;
        }

        aux_vec_idxs.push_back(idx);
        preserved_vec_idxs.push_back(idx);
    }
    OPENVINO_ASSERT(aux_vecs_count() <= aux_vec_idxs.size(), "Failed to allocate required number of vector registers");

    // FPR REGISTERS //
    for (auto idx : pool_fp_gpr_idxs) {
        aux_fp_gpr_idxs.push_back(idx);
    }

    for (size_t idx = 0; idx < get_max_fp_gpr_count(); idx++) {
        if (aux_fp_gpr_idxs.size() >= aux_fp_gprs_count()) {
            break;
        }

        if (std::find(aux_fp_gpr_idxs.begin(), aux_fp_gpr_idxs.end(), idx) != aux_fp_gpr_idxs.end()) {
            continue;
        }
        if (std::find(preserved_fp_gpr_idxs.begin(), preserved_fp_gpr_idxs.end(), idx) != preserved_fp_gpr_idxs.end()) {
            continue;
        }

        aux_fp_gpr_idxs.push_back(idx);
        preserved_fp_gpr_idxs.push_back(idx);
    }
    OPENVINO_ASSERT(aux_fp_gprs_count() <= aux_fp_gpr_idxs.size(), "Failed to allocate required number of FP registers");

    // INT REGISTERS //
    for (auto idx : pool_gpr_idxs) {
        aux_gpr_idxs.push_back(idx);
    }

    const std::unordered_set<size_t> blacklist_gpr = {
        Xbyak_riscv::zero.getIdx(), Xbyak_riscv::ra.getIdx(), Xbyak_riscv::sp.getIdx(), Xbyak_riscv::gp.getIdx(), Xbyak_riscv::tp.getIdx()
    };

    const size_t last_gpr_idx = x31.getIdx();
    for (size_t gpr_idx = 0; gpr_idx <= last_gpr_idx; ++gpr_idx) {
        size_t _idx = last_gpr_idx - gpr_idx;  // we allocate from the end

        if (aux_gpr_idxs.size() >= aux_gprs_count()) {
            break;
        }
        if (blacklist_gpr.count(_idx) > 0) {
            continue;
        }
        if (!is_vec_input) {
            if (std::find(in_idxs.begin(), in_idxs.end(), _idx) != in_idxs.end()) {
                continue;
            }
        }
        if (!is_vec_output) {
            if (std::find(out_idxs.begin(), out_idxs.end(), _idx) != out_idxs.end()) {
                continue;
            }
        }
        if (std::find(aux_gpr_idxs.begin(), aux_gpr_idxs.end(), _idx) != aux_gpr_idxs.end()) {
            continue;
        }
        if (std::find(preserved_gpr_idxs.begin(), preserved_gpr_idxs.end(), _idx) != preserved_gpr_idxs.end()) {
            continue;
        }

        aux_gpr_idxs.push_back(_idx);
        preserved_gpr_idxs.push_back(_idx);
    }
    OPENVINO_ASSERT(aux_gprs_count() <= aux_gpr_idxs.size(), "Failed to allocate required number of general-purpose registers");

    if (!entry_map_.empty()) {
        // last aux_gpr_idx is for p_table, we can use aux_gpr_idxs from idx 0 for other purpose
        p_table = Reg(aux_gpr_idxs[aux_gprs_count() - 1]);
        aux_gpr_idxs.erase(aux_gpr_idxs.end() - 1);
    }

    store_context(preserved_gpr_idxs, preserved_fp_gpr_idxs, preserved_vec_idxs);

    if (!entry_map_.empty()) {
        load_table_addr();
    }
}

void jit_emitter::emitter_postamble() const {
    restore_context(preserved_gpr_idxs, preserved_fp_gpr_idxs, preserved_vec_idxs);

    preserved_vec_idxs.clear();
    preserved_gpr_idxs.clear();
    preserved_fp_gpr_idxs.clear();

    aux_vec_idxs.clear();
    aux_gpr_idxs.clear();
    aux_fp_gpr_idxs.clear();
}

void jit_emitter::store_context(const std::vector<size_t>& gpr_regs,
                                const std::vector<size_t>& fp_gpr_regs,
                                const std::vector<size_t>& vec_regs,
                                const std::unordered_set<size_t>& ignore_vec_regs) const {
    std::vector<size_t> target_vec_regs;
    std::set_difference(vec_regs.begin(), vec_regs.end(), ignore_vec_regs.begin(), ignore_vec_regs.end(),
                        std::inserter(target_vec_regs, target_vec_regs.begin()));

    // GPRs
    {
        const auto gpr_all_size = gpr_regs.size() * get_gpr_length();
        const int frame_size = rnd_up(gpr_all_size, sp_aligment);
        h->addi(sp, sp, -frame_size);
        int imm = 0;
        for (const auto& gpr_idx : gpr_regs) {
            h->sd(Reg(gpr_idx), sp, imm);
            imm += get_gpr_length();
        }
    }

    // FPs
    {
        const auto fp_gpr_all_size = fp_gpr_regs.size() * get_fp_gpr_length();
        const int frame_size = rnd_up(fp_gpr_all_size, sp_aligment);
        h->addi(sp, sp, -frame_size);
        int imm = 0;
        for (const auto& fp_gpr_idx : fp_gpr_regs) {
            h->fsd(FReg(fp_gpr_idx), sp, imm);
            imm += get_fp_gpr_length();
        }
    }

    // Vec regs
    {
        // TODO: support lmul
        const int step = -rnd_up(get_vec_length(), sp_aligment);
        for (const auto& vec_idx : target_vec_regs) {
            h->addi(sp, sp, step);
            h->vse32_v(VReg(vec_idx), sp);
        }
    }
}

void jit_emitter::restore_context(const std::vector<size_t>& gpr_regs,
                                  const std::vector<size_t>& fp_gpr_regs,
                                  const std::vector<size_t>& vec_regs,
                                  const std::unordered_set<size_t>& ignore_vec_regs) const {
    std::vector<size_t> target_vec_regs;
    std::set_difference(vec_regs.begin(), vec_regs.end(), ignore_vec_regs.begin(), ignore_vec_regs.end(),
                        std::inserter(target_vec_regs, target_vec_regs.begin()));

    // GPRs
    {
        const auto gpr_all_size = gpr_regs.size() * get_gpr_length();
        const int frame_size = rnd_up(gpr_all_size, sp_aligment);
        int imm = 0;
        for (const auto& gpr_idx : gpr_regs) {
            h->ld(Reg(gpr_idx), sp, imm);
            imm += get_gpr_length();
        }
        h->addi(sp, sp, frame_size);
    }

    // FPs
    {
        const auto fp_gpr_all_size = fp_gpr_regs.size() * get_fp_gpr_length();
        const int frame_size = rnd_up(fp_gpr_all_size, sp_aligment);
        int imm = 0;
        for (const auto& fp_gpr_idx : fp_gpr_regs) {
            h->fld(FReg(fp_gpr_idx), sp, imm);
            imm += get_fp_gpr_length();
        }
        h->addi(sp, sp, frame_size);
    }

    // Vec regs
    {
        // TODO: support lmul
        const int step = rnd_up(get_vec_length(), sp_aligment);
        for (const auto& vec_idx : target_vec_regs) {
            h->addi(sp, sp, step);
            h->vle32_v(VReg(vec_idx), sp);
        }
    }
}

void jit_emitter::prepare_table() {
    register_table_entries();

    // Now that we registered the entries, we set the offsets.  No
    // entries should be registered after this point.  This allows to
    // expect the same order when injecting the table entries in
    // prepare_table.
    size_t off = 0;
    for (auto it = entry_map_.begin(); it != entry_map_.end(); it++) {
        auto& te = (*it).second;
        te.off = off;
        off += sizeof(table_entry_val_t);
    }
}

void jit_emitter::fill_table(table_entry_val_t* tbl, size_t size) const {
    OPENVINO_ASSERT(entry_map_.size() == size, "Incorrect table size for filling");
    for (const auto& te : entry_map_) {
        const auto& elem = te.second;
        const auto idx = elem.off / sizeof(table_entry_val_t);
        OPENVINO_ASSERT(idx < size, "Incorrect index of table");
        tbl[idx] = elem.val;
    }
}

}  // namespace riscv64
}  // namespace intel_cpu
}  // namespace ov
