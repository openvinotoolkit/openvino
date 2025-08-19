// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <set>
#include <vector>

#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "snippets/emitter.hpp"
#include "snippets/lowered/expression_port.hpp"
#include "xbyak_riscv/xbyak_riscv.hpp"

namespace ov::intel_cpu::riscv64::utils {

inline static std::vector<Xbyak_riscv::Reg> transform_idxs_to_regs(const std::vector<size_t>& idxs) {
    std::vector<Xbyak_riscv::Reg> regs;
    regs.reserve(idxs.size());
    std::transform(idxs.begin(), idxs.end(), std::back_inserter(regs), [](size_t idx) {
        return Xbyak_riscv::Reg(static_cast<int>(idx));
    });
    return regs;
}

/**
 * @brief Find the available register from the pool excepting: a0, a1, sp, ra and `used_gpr_idxs`
 * @param used_gpr_idxs current used gpr register indexes
 * @return register
 */
Xbyak_riscv::Reg get_aux_gpr(const std::vector<size_t>& used_gpr_idxs);

/**
 * @brief Find multiple available registers from the pool excepting: a0, a1, sp, ra and `used_gpr_idxs`
 * @param used_gpr_idxs current used gpr register indexes
 * @param count number of auxiliary registers needed (default: 3)
 * @return vector of registers
 */
std::vector<Xbyak_riscv::Reg> get_aux_gprs(const std::vector<size_t>& used_gpr_idxs, size_t count = 3);

/**
 * @brief Returns an auxiliary GPR register. Returns a register from `aux_gpr_idxs`.
 * If it's empty, then choose a register that is not in `used_gpr_reg_idxs` and add it to `regs_to_spill`.
 * @param used_gpr_reg_idxs register indexes reserved to store memory pointers in this emitter
 * @param aux_gpr_idxs pool of available gp register indexes
 * @param regs_to_spill set of live registers to be spilled before ABI call
 */
Xbyak_riscv::Reg init_memory_access_aux_gpr(const std::vector<size_t>& used_gpr_reg_idxs,
                                             const std::vector<size_t>& aux_gpr_idxs,
                                             std::set<snippets::Reg>& regs_to_spill);

/**
 * @brief Push data pointer on stack adding offset. The offset is taken from runtime params `a0`
 * @param h generator
 * @param stack_offset stack offset
 * @param ptr_reg register containing data pointer
 * @param aux_regs vector of available auxiliary registers (must contain >= 3 registers, ptr_reg must not be in this
 * vector)
 * @param runtime_offset offset in runtime params `a0`
 */
void push_ptr_with_runtime_offset_on_stack(ov::intel_cpu::riscv64::jit_generator_t* h,
                                           int32_t stack_offset,
                                           const Xbyak_riscv::Reg& ptr_reg,
                                           const std::vector<Xbyak_riscv::Reg>& aux_regs,
                                           size_t runtime_offset);

/**
 * @brief Push data pointer on stack adding static offset `ptr_offset`
 * Note: This helper doesn't allocate stack space - the user should guarantee allocated space on stack
 * @param h generator
 * @param stack_offset stack offset
 * @param ptr_reg register containing data pointer
 * @param aux_regs vector of available auxiliary registers (must contain >= 2 registers, ptr_reg must not be in this
 * vector)
 * @param ptr_offset offset which will be added to data pointer
 */
void push_ptr_with_static_offset_on_stack(ov::intel_cpu::riscv64::jit_generator_t* h,
                                          int32_t stack_offset,
                                          const Xbyak_riscv::Reg& ptr_reg,
                                          const std::vector<Xbyak_riscv::Reg>& aux_regs,
                                          size_t ptr_offset);

}  // namespace ov::intel_cpu::riscv64::utils