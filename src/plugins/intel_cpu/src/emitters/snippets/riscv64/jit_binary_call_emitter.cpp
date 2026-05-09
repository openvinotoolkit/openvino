// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_binary_call_emitter.hpp"

#include <algorithm>
#include <cstddef>
#include <set>
#include <utility>
#include <vector>

#include "emitters/plugin/riscv64/jit_context_helpers.hpp"
#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "nodes/kernels/riscv64/cpu_isa_traits.hpp"
#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "openvino/core/except.hpp"
#include "snippets/emitter.hpp"
#include "utils/general_utils.h"
#include "xbyak_riscv/xbyak_riscv.hpp"

namespace ov::intel_cpu::riscv64 {

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
    utils::sub_sp(*h, vector_frame_size);
    utils::save_vector_state(*h, Xbyak_riscv::t0, Xbyak_riscv::t1, 0, get_gpr_length());
    utils::save_vregs(*h, Xbyak_riscv::t0, Xbyak_riscv::t1, vector_state_bytes, vec_regs);
}

void jit_binary_call_emitter::binary_call_postamble() const {
    OPENVINO_ASSERT(m_regs_initialized, "Binary call registers must be initialized first");

    const auto vec_regs = get_vec_regs_to_spill();
    if (!vec_regs.empty()) {
        const auto vector_state_bytes = 2 * get_gpr_length();
        const auto vector_frame_size = rnd_up(vector_state_bytes + vec_regs.size() * get_vec_length(), sp_alignment);

        utils::restore_vregs(*h, Xbyak_riscv::t0, Xbyak_riscv::t1, vector_state_bytes, vec_regs);
        utils::restore_vector_state(*h, Xbyak_riscv::t0, Xbyak_riscv::t1, 0, get_gpr_length());
        utils::add_sp(*h, vector_frame_size);
    }

    restore_context(get_gpr_regs_to_spill(), {}, {});
}

}  // namespace ov::intel_cpu::riscv64
