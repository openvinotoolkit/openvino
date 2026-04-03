// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_binary_call_emitter.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "nodes/kernels/riscv64/cpu_isa_traits.hpp"
#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "openvino/core/except.hpp"
#include "snippets/emitter.hpp"
#include "utils/general_utils.h"
#include "xbyak_riscv/xbyak_riscv.hpp"
#include "xbyak_riscv/xbyak_riscv_csr.hpp"

namespace ov::intel_cpu::riscv64 {

namespace {

void adjust_stack(jit_generator_t* h, size_t bytes, bool allocate) {
    while (bytes > 0) {
        const auto chunk = static_cast<int32_t>(std::min<size_t>(bytes, 2047));
        h->addi(Xbyak_riscv::sp, Xbyak_riscv::sp, allocate ? -chunk : chunk);
        bytes -= chunk;
    }
}

}  // namespace

jit_binary_call_emitter::jit_binary_call_emitter(jit_generator_t* h, cpu_isa_t isa, std::set<snippets::Reg> live_regs)
    : jit_emitter(h, isa),
      m_live_regs(std::move(live_regs)) {}

void jit_binary_call_emitter::init_binary_call_regs(size_t num_binary_args,
                                                    const std::vector<size_t>& used_gpr_idxs) const {
    if (m_regs_initialized) {
        return;
    }

    OPENVINO_ASSERT(
        num_binary_args <= sizeof(jit_generator_t::abi_param_regs) / sizeof(jit_generator_t::abi_param_regs[0]),
        "Requested number of runtime arguments is not supported");

    std::vector<size_t> reserved_regs = used_gpr_idxs;
    for (size_t i = 0; i < num_binary_args; ++i) {
        reserved_regs.push_back(jit_generator_t::abi_param_regs[i].getIdx());
    }

    const auto found_reg = std::find_if(aux_gpr_idxs.rbegin(), aux_gpr_idxs.rend(), [&reserved_regs](size_t idx) {
        return std::find(reserved_regs.begin(), reserved_regs.end(), idx) == reserved_regs.end();
    });
    OPENVINO_ASSERT(found_reg != aux_gpr_idxs.rend(), "Failed to allocate a call-address register");

    m_call_address_reg = Xbyak_riscv::Reg(static_cast<int>(*found_reg));
    m_regs_initialized = true;
}

void jit_binary_call_emitter::init_binary_call_regs(size_t num_binary_args,
                                                    const std::vector<size_t>& in,
                                                    const std::vector<size_t>& out) const {
    std::vector<size_t> used_gpr_idxs = in;
    used_gpr_idxs.insert(used_gpr_idxs.end(), out.begin(), out.end());
    init_binary_call_regs(num_binary_args, used_gpr_idxs);
}

const Xbyak_riscv::Reg& jit_binary_call_emitter::get_call_address_reg() const {
    OPENVINO_ASSERT(m_regs_initialized, "Binary call registers must be initialized first");
    return m_call_address_reg;
}

std::vector<size_t> jit_binary_call_emitter::get_gpr_regs_to_spill() const {
    std::vector<size_t> regs;
    regs.reserve(m_live_regs.size());
    for (const auto& reg : m_live_regs) {
        if (reg.type == snippets::RegType::gpr) {
            regs.push_back(reg.idx);
        }
    }
    return regs;
}

std::vector<size_t> jit_binary_call_emitter::get_vec_regs_to_spill() const {
    std::vector<size_t> regs;
    regs.reserve(m_live_regs.size());
    for (const auto& reg : m_live_regs) {
        if (reg.type == snippets::RegType::vec || reg.type == snippets::RegType::mask) {
            regs.push_back(reg.idx);
        }
    }
    return regs;
}

void jit_binary_call_emitter::binary_call_preamble() const {
    OPENVINO_ASSERT(m_regs_initialized, "Binary call registers must be initialized first");

    store_context(get_gpr_regs_to_spill(), {}, {});

    const auto vec_regs = get_vec_regs_to_spill();
    if (vec_regs.empty()) {
        return;
    }

    const auto vector_state_bytes = 2 * get_gpr_length();
    const auto vector_frame_size = rnd_up(vector_state_bytes + vec_regs.size() * get_vec_length(), sp_alignment);
    adjust_stack(h, vector_frame_size, true);

    const auto saved_vl = Xbyak_riscv::t0;
    const auto saved_vtype = Xbyak_riscv::t1;
    h->csrr(saved_vl, Xbyak_riscv::CSR::vl);
    h->csrr(saved_vtype, Xbyak_riscv::CSR::vtype);
    h->sd(saved_vl, Xbyak_riscv::sp, 0);
    h->sd(saved_vtype, Xbyak_riscv::sp, static_cast<int32_t>(get_gpr_length()));

    h->uni_li(saved_vl, get_vec_length());
    h->vsetvli(Xbyak_riscv::zero, saved_vl, Xbyak_riscv::SEW::e8, Xbyak_riscv::LMUL::m1);

    h->addi(saved_vtype, Xbyak_riscv::sp, static_cast<int32_t>(vector_state_bytes));
    for (const auto& vec_idx : vec_regs) {
        h->vse8_v(Xbyak_riscv::VReg(static_cast<int>(vec_idx)), saved_vtype);
        h->add(saved_vtype, saved_vtype, saved_vl);
    }
}

void jit_binary_call_emitter::binary_call_postamble() const {
    OPENVINO_ASSERT(m_regs_initialized, "Binary call registers must be initialized first");

    const auto vec_regs = get_vec_regs_to_spill();
    if (!vec_regs.empty()) {
        const auto vector_state_bytes = 2 * get_gpr_length();
        const auto vector_frame_size = rnd_up(vector_state_bytes + vec_regs.size() * get_vec_length(), sp_alignment);

        const auto saved_vl = Xbyak_riscv::t0;
        const auto saved_vtype = Xbyak_riscv::t1;
        h->uni_li(saved_vl, get_vec_length());
        h->vsetvli(Xbyak_riscv::zero, saved_vl, Xbyak_riscv::SEW::e8, Xbyak_riscv::LMUL::m1);

        h->addi(saved_vtype, Xbyak_riscv::sp, static_cast<int32_t>(vector_state_bytes));
        for (const auto& vec_idx : vec_regs) {
            h->vle8_v(Xbyak_riscv::VReg(static_cast<int>(vec_idx)), saved_vtype);
            h->add(saved_vtype, saved_vtype, saved_vl);
        }

        h->ld(saved_vl, Xbyak_riscv::sp, 0);
        h->ld(saved_vtype, Xbyak_riscv::sp, static_cast<int32_t>(get_gpr_length()));
        h->vsetvl(Xbyak_riscv::zero, saved_vl, saved_vtype);
        adjust_stack(h, vector_frame_size, false);
    }

    restore_context(get_gpr_regs_to_spill(), {}, {});
}

}  // namespace ov::intel_cpu::riscv64
