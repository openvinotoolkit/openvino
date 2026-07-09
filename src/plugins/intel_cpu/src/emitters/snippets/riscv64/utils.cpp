// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <set>
#include <vector>

#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "openvino/core/except.hpp"
#include "snippets/emitter.hpp"
#include "utils/general_utils.h"
#include "xbyak_riscv/xbyak_riscv.hpp"

namespace ov::intel_cpu::riscv64 {

namespace {

constexpr int64_t gpr_size = 8;

void adjust_sp(jit_generator_t* h, int64_t bytes) {
    while (bytes > 0) {
        const auto step = std::min<int64_t>(bytes, 2047);
        h->addi(Xbyak_riscv::sp, Xbyak_riscv::sp, static_cast<int32_t>(step));
        bytes -= step;
    }
    while (bytes < 0) {
        const auto step = std::max<int64_t>(bytes, -2048);
        h->addi(Xbyak_riscv::sp, Xbyak_riscv::sp, static_cast<int32_t>(step));
        bytes -= step;
    }
}

int64_t get_vec_slot_size() {
    return rnd_up(Xbyak_riscv::CPU::getInstance().getVlen() / 8, 16);
}

std::vector<snippets::Reg> filter_regs(const std::vector<snippets::Reg>& regs, snippets::RegType type) {
    std::vector<snippets::Reg> filtered;
    filtered.reserve(regs.size());
    std::copy_if(regs.begin(), regs.end(), std::back_inserter(filtered), [type](const snippets::Reg& reg) {
        return reg.type == type;
    });
    return filtered;
}

void validate_supported_regs(const std::vector<snippets::Reg>& regs) {
    for (const auto& reg : regs) {
        if (reg.type != snippets::RegType::gpr && reg.type != snippets::RegType::vec) {
            OPENVINO_THROW("Unsupported register type in RV64 RegSpill emitter");
        }
    }
}

}  // namespace

EmitABIRegSpills::EmitABIRegSpills(jit_generator_t* h_arg) : h(h_arg) {}

EmitABIRegSpills::~EmitABIRegSpills() {
    OPENVINO_ASSERT(spill_status, "postamble or preamble is missed");
}

void EmitABIRegSpills::store_regs_to_stack(jit_generator_t* h, const std::vector<snippets::Reg>& regs_to_store) {
    validate_supported_regs(regs_to_store);

    const auto vec_slot_size = get_vec_slot_size();
    for (const auto& reg : filter_regs(regs_to_store, snippets::RegType::vec)) {
        adjust_sp(h, -vec_slot_size);
        h->vs1r_v(Xbyak_riscv::VReg(static_cast<int>(reg.idx)), Xbyak_riscv::sp);
    }

    const auto gpr_regs = filter_regs(regs_to_store, snippets::RegType::gpr);
    const auto gpr_frame_size = rnd_up(gpr_regs.size() * gpr_size, 16);
    if (gpr_frame_size > 0) {
        adjust_sp(h, -static_cast<int64_t>(gpr_frame_size));
    }

    int32_t offset = 0;
    for (const auto& reg : gpr_regs) {
        h->sd(Xbyak_riscv::Reg(static_cast<int>(reg.idx)), Xbyak_riscv::sp, offset);
        offset += gpr_size;
    }
}

void EmitABIRegSpills::load_regs_from_stack(jit_generator_t* h, const std::vector<snippets::Reg>& regs_to_load) {
    validate_supported_regs(regs_to_load);

    const auto gpr_regs = filter_regs(regs_to_load, snippets::RegType::gpr);
    int32_t offset = 0;
    for (const auto& reg : gpr_regs) {
        h->ld(Xbyak_riscv::Reg(static_cast<int>(reg.idx)), Xbyak_riscv::sp, offset);
        offset += gpr_size;
    }

    const auto gpr_frame_size = rnd_up(gpr_regs.size() * gpr_size, 16);
    if (gpr_frame_size > 0) {
        adjust_sp(h, static_cast<int64_t>(gpr_frame_size));
    }

    const auto vec_regs = filter_regs(regs_to_load, snippets::RegType::vec);
    const auto vec_slot_size = get_vec_slot_size();
    for (auto it = vec_regs.rbegin(); it != vec_regs.rend(); ++it) {
        h->vl1re8_v(Xbyak_riscv::VReg(static_cast<int>(it->idx)), Xbyak_riscv::sp);
        adjust_sp(h, vec_slot_size);
    }
}

void EmitABIRegSpills::preamble(const std::vector<snippets::Reg>& live_regs) {
    OPENVINO_ASSERT(spill_status, "Attempt to spill ABI registers twice in a row");
    m_regs_to_spill = live_regs;
    store_regs_to_stack(h, m_regs_to_spill);
    spill_status = false;
}

void EmitABIRegSpills::preamble(const std::set<snippets::Reg>& live_regs) {
    preamble(std::vector<snippets::Reg>(live_regs.begin(), live_regs.end()));
}

void EmitABIRegSpills::postamble() {
    OPENVINO_ASSERT(!spill_status, "Attempt to restore ABI registers that were not spilled");
    load_regs_from_stack(h, m_regs_to_spill);
    m_regs_to_spill.clear();
    spill_status = true;
}

}  // namespace ov::intel_cpu::riscv64

namespace ov::intel_cpu::riscv64::utils {

jit_aux_gpr_holder::jit_aux_gpr_holder(ov::intel_cpu::riscv64::jit_generator_t* host,
                                       std::vector<size_t>& pool_gpr_idxs,
                                       const std::vector<size_t>& used_gpr_idxs)
    : m_h(host),
      m_pool_gpr_idxs(pool_gpr_idxs) {
    if (m_pool_gpr_idxs.empty()) {
        // choose an available caller-saved reg not in used set
        m_reg = ov::intel_cpu::riscv64::utils::get_aux_gpr(used_gpr_idxs);
        m_preserved = true;
        // Maintain 16-byte alignment; reserve 16 bytes and save at 0
        m_h->addi(Xbyak_riscv::sp, Xbyak_riscv::sp, -16);
        m_h->sd(m_reg, Xbyak_riscv::sp, 0);
    } else {
        m_reg = Xbyak_riscv::Reg(static_cast<int>(m_pool_gpr_idxs.back()));
        m_pool_gpr_idxs.pop_back();
    }
}

jit_aux_gpr_holder::~jit_aux_gpr_holder() {
    if (m_preserved) {
        m_h->ld(m_reg, Xbyak_riscv::sp, 0);
        m_h->addi(Xbyak_riscv::sp, Xbyak_riscv::sp, 16);
    } else {
        m_pool_gpr_idxs.push_back(static_cast<size_t>(m_reg.getIdx()));
    }
}

Xbyak_riscv::Reg get_aux_gpr(const std::vector<size_t>& used_gpr_idxs) {
    // RISC-V reserved registers to avoid: x0(zero), x1(ra), x2(sp), x3(gp), x4(tp), x8(s0/fp)
    // Also avoid a0, a1 which are used for ABI parameters
    const std::set<size_t> reserved_regs = {0, 1, 2, 3, 4, 8, 10, 11};

    // Start with temporary registers t0-t6 (x5-x7, x28-x31)
    const std::vector<size_t> temp_regs = {5, 6, 7, 28, 29, 30, 31};

    for (size_t reg_idx : temp_regs) {
        if (std::find(used_gpr_idxs.begin(), used_gpr_idxs.end(), reg_idx) == used_gpr_idxs.end()) {
            return Xbyak_riscv::Reg(static_cast<int>(reg_idx));
        }
    }

    // If no temporary registers available, try saved registers s1-s11 (x9, x18-x27)
    const std::vector<size_t> saved_regs = {9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};

    for (size_t reg_idx : saved_regs) {
        if (std::find(used_gpr_idxs.begin(), used_gpr_idxs.end(), reg_idx) == used_gpr_idxs.end()) {
            return Xbyak_riscv::Reg(static_cast<int>(reg_idx));
        }
    }

    OPENVINO_THROW("No available auxiliary GPR registers");
}

}  // namespace ov::intel_cpu::riscv64::utils
