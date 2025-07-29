// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <algorithm>
#include <common/utils.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "snippets/emitter.hpp"
#include "snippets/lowered/expression_port.hpp"
#include "snippets/lowered/expressions/buffer_expression.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/utils/utils.hpp"

using namespace dnnl::impl::cpu::aarch64;

namespace ov::intel_cpu::aarch64::utils {

Xbyak_aarch64::XReg get_aux_gpr(const std::vector<size_t>& used_gpr_idxs) {
    // SP - stack pointer should be preserved, X0 and X1 - runtime parameter registers in the kernel
    // X18 - platform register should not be used
    static std::unordered_set<size_t> blacklist_gpr_idxs = {
        31,  // Stack pointer (SP)
        0,   // abi_param1 (X0)
        1,   // abi_param2 (X1)
        18   // Platform register (X18)
    };

    // Iterate through available GPR registers (X0-X30, excluding X31 which is SP)
    for (size_t gpr_idx = 0; gpr_idx <= 30; ++gpr_idx) {
        size_t _idx = 30 - gpr_idx;  // we allocate from the end
        if (std::find(used_gpr_idxs.cbegin(), used_gpr_idxs.cend(), _idx) != used_gpr_idxs.cend()) {
            continue;
        }
        if (blacklist_gpr_idxs.count(_idx) > 0) {
            continue;
        }
        return Xbyak_aarch64::XReg(_idx);
    }
    OV_CPU_JIT_EMITTER_THROW("Failed to allocate aux GPR");
}

std::vector<Xbyak_aarch64::XReg> get_aux_gprs(const std::vector<size_t>& used_gpr_idxs, size_t count) {
    std::vector<Xbyak_aarch64::XReg> aux_regs;
    aux_regs.reserve(count);
    std::vector<size_t> temp_used_indices = used_gpr_idxs;

    for (size_t i = 0; i < count; i++) {
        auto aux_reg = get_aux_gpr(temp_used_indices);
        aux_regs.push_back(aux_reg);
        temp_used_indices.push_back(aux_reg.getIdx());
    }

    return aux_regs;
}

Xbyak_aarch64::XReg init_memory_access_aux_gpr(const std::vector<size_t>& used_gpr_reg_idxs,
                                               const std::vector<size_t>& aux_gpr_idxs,
                                               std::set<snippets::Reg>& regs_to_spill) {
    if (!aux_gpr_idxs.empty()) {
        return Xbyak_aarch64::XReg(static_cast<int>(aux_gpr_idxs[0]));
    }
    const auto aux_reg = get_aux_gpr(used_gpr_reg_idxs);
    regs_to_spill.emplace(snippets::RegType::gpr, aux_reg.getIdx());
    return aux_reg;
}

void push_ptr_with_runtime_offset_on_stack(dnnl::impl::cpu::aarch64::jit_generator* h,
                                           int32_t stack_offset,
                                           const Xbyak_aarch64::XReg& ptr_reg,
                                           const std::vector<Xbyak_aarch64::XReg>& aux_regs,
                                           size_t runtime_offset) {
    // Safety assertions as suggested
    OV_CPU_JIT_EMITTER_ASSERT(aux_regs.size() >= 3, "aux_regs must contain at least 3 registers");

    // Assert that ptr_reg is not in aux_regs
    for (const auto& reg : aux_regs) {
        OV_CPU_JIT_EMITTER_ASSERT(reg.getIdx() != ptr_reg.getIdx(), "ptr_reg must not be in aux_regs");
    }

    // Use safe auxiliary registers from the provided set
    const Xbyak_aarch64::XReg aux_reg = aux_regs[0];   // For storing adjusted pointer
    const Xbyak_aarch64::XReg temp_reg = aux_regs[1];  // For temporary calculations
    const Xbyak_aarch64::XReg addr_reg = aux_regs[2];  // For address calculations in add_imm

    // Copy pointer to aux register
    h->mov(aux_reg, ptr_reg);

    // Load the runtime offset from abi_param1 (X0) and add it to the pointer
    Xbyak_aarch64::XReg abi_param1(0);

    // Load the offset value from the runtime parameter location
    h->add_imm(temp_reg, abi_param1, runtime_offset, addr_reg);
    h->ldr(temp_reg, Xbyak_aarch64::ptr(temp_reg));

    h->add(aux_reg, aux_reg, temp_reg);

    // Store the adjusted pointer on stack
    h->str(aux_reg, Xbyak_aarch64::ptr(h->sp, stack_offset));
}

void push_ptr_with_static_offset_on_stack(dnnl::impl::cpu::aarch64::jit_generator* h,
                                          int32_t stack_offset,
                                          const Xbyak_aarch64::XReg& ptr_reg,
                                          const std::vector<Xbyak_aarch64::XReg>& aux_regs,
                                          size_t ptr_offset) {
    // If there's no static offset, just store the pointer
    if (ptr_offset == 0) {
        h->str(ptr_reg, Xbyak_aarch64::ptr(h->sp, stack_offset));
        return;
    }

    // Safety assertions as suggested
    OV_CPU_JIT_EMITTER_ASSERT(aux_regs.size() >= 2, "aux_regs must contain at least 2 registers");

    // Assert that ptr_reg is not in aux_regs
    for (const auto& reg : aux_regs) {
        OV_CPU_JIT_EMITTER_ASSERT(reg.getIdx() != ptr_reg.getIdx(), "ptr_reg must not be in aux_regs");
    }

    // Use safe auxiliary registers from the provided vector
    const Xbyak_aarch64::XReg temp_reg = aux_regs[0];  // For storing adjusted pointer
    const Xbyak_aarch64::XReg addr_reg = aux_regs[1];  // For address calculations in add_imm

    // For non-zero offsets, apply the offset and then store
    h->add_imm(temp_reg, ptr_reg, ptr_offset, addr_reg);

    // Store the adjusted pointer on stack
    h->str(temp_reg, Xbyak_aarch64::ptr(h->sp, stack_offset));
}

void push_and_load_ptrs_with_offsets(dnnl::impl::cpu::aarch64::jit_generator* h,
                                     const std::vector<Xbyak_aarch64::XReg>& mem_ptrs,
                                     const std::vector<size_t>& memory_offsets,
                                     const std::vector<size_t>& buffer_ids,
                                     const std::vector<Xbyak_aarch64::XReg>& aux_regs,
                                     const std::vector<Xbyak_aarch64::XReg>& load_regs) {
    const size_t gpr_length = 8;     // 64-bit register length
    const size_t sp_alignment = 16;  // AArch64 stack alignment requirement

    // Allocate stack space for all pointers
    const auto sp_size = rnd_up(mem_ptrs.size() * gpr_length, sp_alignment);
    h->sub(h->sp, h->sp, sp_size);

    // Push all pointers with offsets onto stack
    for (size_t i = 0; i < mem_ptrs.size(); i++) {
        const auto& ptr_reg = mem_ptrs[i];
        int32_t stack_offset = i * gpr_length;

        if (ov::snippets::utils::is_dynamic_value(memory_offsets[i])) {
            // Dynamic offset: read from runtime parameters
            size_t runtime_offset = GET_OFF(buffer_offsets) + buffer_ids[i] * sizeof(size_t);
            push_ptr_with_runtime_offset_on_stack(h, stack_offset, ptr_reg, aux_regs, runtime_offset);
        } else {
            // Static offset: add compile-time constant
            push_ptr_with_static_offset_on_stack(h, stack_offset, ptr_reg, aux_regs, memory_offsets[i]);
        }
    }

    // Load back the adjusted pointers to specified registers
    for (size_t i = 0; i < load_regs.size() && i < mem_ptrs.size(); i++) {
        h->ldr(load_regs[i], Xbyak_aarch64::ptr(h->sp, static_cast<int32_t>(i * gpr_length)));
    }

    // Restore stack pointer
    h->add(h->sp, h->sp, sp_size);
}

}  // namespace ov::intel_cpu::aarch64::utils
