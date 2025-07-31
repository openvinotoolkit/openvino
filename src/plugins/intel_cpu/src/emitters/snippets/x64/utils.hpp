// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xbyak/xbyak.h>

#include <algorithm>
#include <cstddef>
#include <set>
#include <vector>

#include "cpu/x64/jit_generator.hpp"
#include "snippets/emitter.hpp"
#include "snippets/lowered/expression_port.hpp"

namespace ov::intel_cpu::utils {

inline static std::vector<Xbyak::Reg64> transform_idxs_to_regs(const std::vector<size_t>& idxs) {
    std::vector<Xbyak::Reg64> regs(idxs.size());
    std::transform(idxs.begin(), idxs.end(), regs.begin(), [](size_t idx) {
        return Xbyak::Reg64(static_cast<int>(idx));
    });
    return regs;
}

/**
 * @brief Find the available register from the pool excepting: abi_param1, abi_param2, RSP and `used_gpr_idxs`
 * @param used_gpr_idxs current used gpr register indexes
 * @return register
 */
Xbyak::Reg64 get_aux_gpr(const std::vector<size_t>& used_gpr_idxs);

/**
 * @brief Returns aux gpr register for dynamic memory access emitters. Returns a register from `aux_gpr_idxs`.
 * If it's empty, then choose a register that is not in `mem_ptr_reg_idxs` and add it to `regs_to_spill`.
 * @param mem_ptr_reg_idxs register indexes reserved to store memory pointers in this emitter
 * @param aux_gpr_idxs pool of available gp register indexes
 * @param regs_to_spill set of live registers to be spilled before ABI call
 */
Xbyak::Reg64 init_memory_access_aux_gpr(const std::vector<size_t>& used_gpr_reg_idxs,
                                        const std::vector<size_t>& aux_gpr_idxs,
                                        std::set<snippets::Reg>& regs_to_spill);

/**
 * @brief Push data pointer on stack adding offset. The offset is taken from runtime params `abi_param1`
 * @param h generator
 * @param stack_offset stack offset
 * @param ptr_reg register containing data pointer
 * @param aux_reg aux register
 * @param runtime_offset offset in runtime params `abi_param1`
 */
void push_ptr_with_runtime_offset_on_stack(dnnl::impl::cpu::x64::jit_generator_t* h,
                                           size_t stack_offset,
                                           Xbyak::Reg64 ptr_reg,
                                           Xbyak::Reg64 aux_reg,
                                           size_t runtime_offset);

/**
 * @brief Push data pointer on stack adding static offset `ptr_offset`
 * Note: This helper doesn't allocate stack space - the user should guarantee allocated space on stack
 * @param h generator
 * @param stack_offset stack offset
 * @param ptr_reg register containing data pointer
 * @param ptr_offset offset which will be added to data pointer
 */
void push_ptr_with_static_offset_on_stack(dnnl::impl::cpu::x64::jit_generator_t* h,
                                          size_t stack_offset,
                                          Xbyak::Reg64 ptr_reg,
                                          size_t ptr_offset);

}  // namespace ov::intel_cpu::utils
