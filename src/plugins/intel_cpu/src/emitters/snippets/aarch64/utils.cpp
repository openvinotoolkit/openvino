// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_adr.h>
#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_reg.h>

#include <algorithm>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/except.hpp"
#include "snippets/emitter.hpp"
#include "snippets/utils/utils.hpp"
#include "utils/general_utils.h"

using namespace dnnl::impl::cpu::aarch64;

namespace ov::intel_cpu::aarch64 {

namespace {

constexpr uint32_t gpr_size = 8;
constexpr uint32_t vec_size = 16;

[[nodiscard]] uint32_t aligned_size(const uint32_t size) {
    if (size == 0) {
        return 0;
    }
    return ((size + jit_emitter::sp_alignment - 1) / jit_emitter::sp_alignment) * jit_emitter::sp_alignment;
}

[[nodiscard]] bool is_gpr(const Xbyak_aarch64::Reg& reg) {
    return reg.isRReg() && reg.getBit() == gpr_size * 8;
}

[[nodiscard]] bool is_vec(const Xbyak_aarch64::Reg& reg) {
    return reg.isVRegSc() && reg.getBit() == vec_size * 8;
}

[[nodiscard]] uint32_t get_total_gpr_shift(const std::vector<Xbyak_aarch64::Reg>& regs) {
    return aligned_size(static_cast<uint32_t>(std::count_if(regs.begin(), regs.end(), is_gpr) * gpr_size));
}

[[nodiscard]] uint32_t get_total_shift(const std::vector<Xbyak_aarch64::Reg>& regs) {
    return get_total_gpr_shift(regs) +
           aligned_size(static_cast<uint32_t>(std::count_if(regs.begin(), regs.end(), is_vec) * vec_size));
}

[[nodiscard]] Xbyak_aarch64::XReg as_gpr(const Xbyak_aarch64::Reg& reg) {
    OV_CPU_JIT_EMITTER_ASSERT(is_gpr(reg), "Expected a 64-bit GPR register in Arm64 spill helper");
    return Xbyak_aarch64::XReg(reg.getIdx());
}

[[nodiscard]] Xbyak_aarch64::QReg as_vec(const Xbyak_aarch64::Reg& reg) {
    OV_CPU_JIT_EMITTER_ASSERT(is_vec(reg), "Expected a 128-bit vector register in Arm64 spill helper");
    return Xbyak_aarch64::QReg(reg.getIdx());
}

template <typename Predicate, typename RegT>
void store_reg_group(dnnl::impl::cpu::aarch64::jit_generator_t* h,
                     const std::vector<Xbyak_aarch64::Reg>& regs,
                     int32_t current_offset,
                     Predicate&& predicate,
                     RegT&& to_reg,
                     const uint32_t reg_byte_size) {
    std::vector<Xbyak_aarch64::Reg> filtered_regs;
    filtered_regs.reserve(regs.size());
    std::copy_if(regs.begin(), regs.end(), std::back_inserter(filtered_regs), std::forward<Predicate>(predicate));

    size_t i = 0;
    for (; i + 1 < filtered_regs.size(); i += 2) {
        h->stp(to_reg(filtered_regs[i]), to_reg(filtered_regs[i + 1]), Xbyak_aarch64::ptr(h->sp, current_offset));
        current_offset += static_cast<int32_t>(2 * reg_byte_size);
    }
    if (i < filtered_regs.size()) {
        h->str(to_reg(filtered_regs[i]), Xbyak_aarch64::ptr(h->sp, current_offset));
    }
}

template <typename Predicate, typename RegT>
void load_reg_group(dnnl::impl::cpu::aarch64::jit_generator_t* h,
                    const std::vector<Xbyak_aarch64::Reg>& regs,
                    int32_t current_offset,
                    Predicate&& predicate,
                    RegT&& to_reg,
                    const uint32_t reg_byte_size) {
    std::vector<Xbyak_aarch64::Reg> filtered_regs;
    filtered_regs.reserve(regs.size());
    std::copy_if(regs.begin(), regs.end(), std::back_inserter(filtered_regs), std::forward<Predicate>(predicate));

    size_t i = 0;
    for (; i + 1 < filtered_regs.size(); i += 2) {
        h->ldp(to_reg(filtered_regs[i]), to_reg(filtered_regs[i + 1]), Xbyak_aarch64::ptr(h->sp, current_offset));
        current_offset += static_cast<int32_t>(2 * reg_byte_size);
    }
    if (i < filtered_regs.size()) {
        h->ldr(to_reg(filtered_regs[i]), Xbyak_aarch64::ptr(h->sp, current_offset));
    }
}

}  // namespace

EmitABIRegSpills::EmitABIRegSpills(jit_generator_t* h_arg) : h(h_arg) {}

EmitABIRegSpills::~EmitABIRegSpills() {
    OPENVINO_ASSERT(spill_status, "postamble or preamble is missed");
}

std::vector<Xbyak_aarch64::Reg> EmitABIRegSpills::get_regs_to_spill(const std::set<snippets::Reg>& live_regs) {
    std::vector<Xbyak_aarch64::Reg> regs_to_spill;
    regs_to_spill.reserve(live_regs.size());
    for (const auto& reg : live_regs) {
        switch (reg.type) {
        case snippets::RegType::gpr:
            regs_to_spill.emplace_back(Xbyak_aarch64::XReg(reg.idx));
            break;
        case snippets::RegType::vec:
            regs_to_spill.emplace_back(Xbyak_aarch64::QReg(reg.idx));
            break;
        default:
            OPENVINO_THROW("Unsupported register type in Arm64 RegSpill emitter");
        }
    }
    return regs_to_spill;
}

void EmitABIRegSpills::store_regs_to_stack(dnnl::impl::cpu::aarch64::jit_generator_t* h,
                                           const std::vector<Xbyak_aarch64::Reg>& regs_to_store) {
    const auto total_shift = get_total_shift(regs_to_store);
    if (total_shift > 0) {
        h->sub(h->sp, h->sp, total_shift);
    }

    store_reg_group(h, regs_to_store, 0, is_gpr, as_gpr, gpr_size);
    store_reg_group(h,
                    regs_to_store,
                    static_cast<int32_t>(get_total_gpr_shift(regs_to_store)),
                    is_vec,
                    as_vec,
                    vec_size);
}

void EmitABIRegSpills::load_regs_from_stack(dnnl::impl::cpu::aarch64::jit_generator_t* h,
                                            const std::vector<Xbyak_aarch64::Reg>& regs_to_load) {
    const auto total_gpr_shift = get_total_gpr_shift(regs_to_load);
    load_reg_group(h, regs_to_load, static_cast<int32_t>(total_gpr_shift), is_vec, as_vec, vec_size);
    load_reg_group(h, regs_to_load, 0, is_gpr, as_gpr, gpr_size);

    const auto total_shift =
        total_gpr_shift +
        aligned_size(static_cast<uint32_t>(std::count_if(regs_to_load.begin(), regs_to_load.end(), is_vec) * vec_size));
    if (total_shift > 0) {
        h->add(h->sp, h->sp, total_shift);
    }
}

void EmitABIRegSpills::preamble(const std::vector<Xbyak_aarch64::Reg>& live_regs) {
    OPENVINO_ASSERT(spill_status, "Attempt to spill ABI registers twice in a row");
    m_regs_to_spill = live_regs;
    store_regs_to_stack(h, m_regs_to_spill);
    spill_status = false;
}

void EmitABIRegSpills::preamble(const std::set<snippets::Reg>& live_regs) {
    preamble(get_regs_to_spill(live_regs));
}

void EmitABIRegSpills::postamble() {
    OPENVINO_ASSERT(!spill_status, "Attempt to restore ABI registers that were not spilled");
    load_regs_from_stack(h, m_regs_to_spill);
    m_regs_to_spill.clear();
    spill_status = true;
}

namespace utils {

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

void push_ptr_with_runtime_offset_on_stack(dnnl::impl::cpu::aarch64::jit_generator_t* h,
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

void push_ptr_with_static_offset_on_stack(dnnl::impl::cpu::aarch64::jit_generator_t* h,
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

void push_and_load_ptrs_with_offsets(dnnl::impl::cpu::aarch64::jit_generator_t* h,
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
    size_t i = 0;
    for (; i + 1 < mem_ptrs.size(); i += 2) {
        const auto off = stack_offsets[i];
        h->ldp(load_regs[i], load_regs[i + 1], Xbyak_aarch64::ptr(h->sp, off));
    }
    if (i < mem_ptrs.size()) {
        h->ldr(load_regs[i], Xbyak_aarch64::ptr(h->sp, stack_offsets[i]));
    }

    // Restore stack pointer
    h->add(h->sp, h->sp, sp_size);
}

void push_ptrs_with_offsets_to_stack(dnnl::impl::cpu::aarch64::jit_generator_t* h,
                                     const std::vector<Xbyak_aarch64::XReg>& mem_ptrs,
                                     const std::vector<size_t>& memory_offsets,
                                     const std::vector<size_t>& buffer_ids,
                                     const std::vector<Xbyak_aarch64::XReg>& aux_regs,
                                     const std::vector<int32_t>& stack_offsets) {
    OV_CPU_JIT_EMITTER_ASSERT(mem_ptrs.size() == memory_offsets.size(), "mem_ptrs and memory_offsets size mismatch");
    OV_CPU_JIT_EMITTER_ASSERT(mem_ptrs.size() == buffer_ids.size(), "mem_ptrs and buffer_ids size mismatch");
    OV_CPU_JIT_EMITTER_ASSERT(mem_ptrs.size() == stack_offsets.size(), "mem_ptrs and stack_offsets size mismatch");

    // Fast path: pair-store original pointers when two consecutive entries do not need adjustment
    std::vector<bool> handled(mem_ptrs.size(), false);
    for (size_t i = 0; i + 1 < mem_ptrs.size(); i += 2) {
        const bool left_static = !ov::snippets::utils::is_dynamic_value(memory_offsets[i]);
        const bool right_static = !ov::snippets::utils::is_dynamic_value(memory_offsets[i + 1]);
        if (left_static && right_static && memory_offsets[i] == 0 && memory_offsets[i + 1] == 0) {
            // stack_offsets are i*8 (contiguous) and SP is 16B aligned ⇒ stp is safe
            h->stp(mem_ptrs[i], mem_ptrs[i + 1], Xbyak_aarch64::ptr(h->sp, stack_offsets[i]));
            handled[i] = handled[i + 1] = true;
        }
    }

    // Store remaining pointers with proper adjustments
    for (size_t i = 0; i < mem_ptrs.size(); i++) {
        if (handled[i]) {
            continue;
        }
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

}  // namespace utils
}  // namespace ov::intel_cpu::aarch64
