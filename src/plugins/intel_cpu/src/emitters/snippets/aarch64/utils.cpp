// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_adr.h>
#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_reg.h>

#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <set>
#include <unordered_set>
#include <vector>

#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/except.hpp"
#include "snippets/emitter.hpp"
#include "snippets/utils/utils.hpp"
#include "utils/general_utils.h"

using namespace dnnl::impl::cpu::aarch64;

namespace ov::intel_cpu::aarch64::utils {

std::vector<Xbyak_aarch64::XReg> get_aux_gprs(const std::vector<size_t>& used_gpr_idxs, size_t count) {
    // X0 and X1 - runtime parameter registers in the kernel
    // X18 - platform register should not be used
    // SP - stack pointer should be preserved
    static const std::unordered_set<size_t> blacklist_gpr_idxs = {
        0,   // abi_param1 (X0)
        1,   // abi_param2 (X1)
        18,  // Platform register (X18)
        31,  // Stack pointer (SP)
    };

    OPENVINO_ASSERT(count <= 32 - blacklist_gpr_idxs.size(),
                    "Cannot allocate more than ",
                    32 - blacklist_gpr_idxs.size(),
                    " auxiliary registers");

    // Convert used_gpr_idxs to unordered_set for O(1) lookups
    const std::unordered_set<size_t> used_set(used_gpr_idxs.begin(), used_gpr_idxs.end());

    std::vector<Xbyak_aarch64::XReg> aux_regs;
    aux_regs.reserve(count);

    // Iterate from X30 down to X0 (allocate from the end)
    for (size_t idx = 30; idx != SIZE_MAX; --idx) {
        if (used_set.count(idx) || blacklist_gpr_idxs.count(idx)) {
            continue;
        }
        aux_regs.emplace_back(idx);
        if (aux_regs.size() == count) {
            break;
        }
    }

    OPENVINO_ASSERT(aux_regs.size() == count, "Expected ", count, " auxiliary registers, but got ", aux_regs.size());
    return aux_regs;
}

Xbyak_aarch64::XReg get_aux_gpr(const std::vector<size_t>& used_gpr_idxs) {
    return get_aux_gprs(used_gpr_idxs, 1)[0];
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
    OV_CPU_JIT_EMITTER_ASSERT(mem_ptrs.size() == load_regs.size(), "mem_ptrs and load_regs size mismatch");

    const size_t gpr_length = 8;     // 64-bit register length
    const size_t sp_alignment = 16;  // AArch64 stack alignment requirement

    // Allocate stack space for all pointers
    const auto sp_size = rnd_up(mem_ptrs.size() * gpr_length, sp_alignment);
    h->sub(h->sp, h->sp, sp_size);

    // Generate stack offsets for sequential storage
    std::vector<int32_t> stack_offsets;
    stack_offsets.reserve(mem_ptrs.size());
    for (size_t i = 0; i < mem_ptrs.size(); i++) {
        stack_offsets.push_back(static_cast<int32_t>(i * gpr_length));
    }

    // Use the common function to push pointers with offsets to stack
    push_ptrs_with_offsets_to_stack(h, mem_ptrs, memory_offsets, buffer_ids, aux_regs, stack_offsets);

    // Load back the adjusted pointers to specified registers
    for (size_t i = 0; i < mem_ptrs.size(); i++) {
        h->ldr(load_regs[i], Xbyak_aarch64::ptr(h->sp, stack_offsets[i]));
    }

    // Restore stack pointer
    h->add(h->sp, h->sp, sp_size);
}

void push_ptrs_with_offsets_to_stack(dnnl::impl::cpu::aarch64::jit_generator* h,
                                     const std::vector<Xbyak_aarch64::XReg>& mem_ptrs,
                                     const std::vector<size_t>& memory_offsets,
                                     const std::vector<size_t>& buffer_ids,
                                     const std::vector<Xbyak_aarch64::XReg>& aux_regs,
                                     const std::vector<int32_t>& stack_offsets) {
    OV_CPU_JIT_EMITTER_ASSERT(mem_ptrs.size() == memory_offsets.size(), "mem_ptrs and memory_offsets size mismatch");
    OV_CPU_JIT_EMITTER_ASSERT(mem_ptrs.size() == buffer_ids.size(), "mem_ptrs and buffer_ids size mismatch");
    OV_CPU_JIT_EMITTER_ASSERT(mem_ptrs.size() == stack_offsets.size(), "mem_ptrs and stack_offsets size mismatch");

    // Store all pointers with offsets to their specific stack locations
    for (size_t i = 0; i < mem_ptrs.size(); i++) {
        const auto& ptr_reg = mem_ptrs[i];
        int32_t stack_offset = stack_offsets[i];

        if (ov::snippets::utils::is_dynamic_value(memory_offsets[i])) {
            OPENVINO_ASSERT(!ov::snippets::utils::is_dynamic_value(buffer_ids[i]),
                            "In dynamic case Buffer ID must be defined");
            // Dynamic offset: read from runtime parameters
            size_t runtime_offset = GET_OFF(buffer_offsets) + buffer_ids[i] * sizeof(size_t);
            push_ptr_with_runtime_offset_on_stack(h, stack_offset, ptr_reg, aux_regs, runtime_offset);
        } else {
            // Static offset: add compile-time constant
            size_t offset = memory_offsets[i];
            push_ptr_with_static_offset_on_stack(h, stack_offset, ptr_reg, aux_regs, offset);
        }
    }
}

}  // namespace ov::intel_cpu::aarch64::utils
