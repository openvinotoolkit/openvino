// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

#include "nodes/kernels/riscv64/jit_generator.hpp"
#include "openvino/core/except.hpp"
#include "snippets/emitter.hpp"
#include "snippets/lowered/expression_port.hpp"
#include "xbyak_riscv/xbyak_riscv.hpp"

namespace ov::intel_cpu::riscv64::utils {

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

std::vector<Xbyak_riscv::Reg> get_aux_gprs(const std::vector<size_t>& used_gpr_idxs, size_t count) {
    std::vector<Xbyak_riscv::Reg> result;
    std::vector<size_t> current_used = used_gpr_idxs;
    
    for (size_t i = 0; i < count; ++i) {
        auto aux_reg = get_aux_gpr(current_used);
        result.push_back(aux_reg);
        current_used.push_back(aux_reg.getIdx());
    }
    
    return result;
}

Xbyak_riscv::Reg init_memory_access_aux_gpr(const std::vector<size_t>& used_gpr_reg_idxs,
                                             const std::vector<size_t>& aux_gpr_idxs,
                                             std::set<snippets::Reg>& regs_to_spill) {
    if (!aux_gpr_idxs.empty()) {
        return Xbyak_riscv::Reg(static_cast<int>(aux_gpr_idxs.front()));
    }
    
    // Find an available register and mark it for spilling
    auto aux_reg = get_aux_gpr(used_gpr_reg_idxs);
    regs_to_spill.insert({snippets::RegType::gpr, aux_reg.getIdx()});
    return aux_reg;
}

void push_ptr_with_runtime_offset_on_stack(ov::intel_cpu::riscv64::jit_generator_t* h,
                                           int32_t stack_offset,
                                           const Xbyak_riscv::Reg& ptr_reg,
                                           const std::vector<Xbyak_riscv::Reg>& aux_regs,
                                           size_t runtime_offset) {
    OPENVINO_ASSERT(aux_regs.size() >= 3, "Need at least 3 auxiliary registers");
    
    auto& aux1 = aux_regs[0];
    auto& aux2 = aux_regs[1];
    
    // Load runtime offset from runtime params
    h->lw(aux1, Xbyak_riscv::a0, static_cast<int32_t>(runtime_offset));
    
    // Add offset to pointer
    h->add(aux2, ptr_reg, aux1);
    
    // Store adjusted pointer to stack
    h->sw(aux2, Xbyak_riscv::sp, stack_offset);
}

void push_ptr_with_static_offset_on_stack(ov::intel_cpu::riscv64::jit_generator_t* h,
                                          int32_t stack_offset,
                                          const Xbyak_riscv::Reg& ptr_reg,
                                          const std::vector<Xbyak_riscv::Reg>& aux_regs,
                                          size_t ptr_offset) {
    OPENVINO_ASSERT(aux_regs.size() >= 2, "Need at least 2 auxiliary registers");
    
    if (ptr_offset == 0) {
        // Direct store without offset
        h->sw(ptr_reg, Xbyak_riscv::sp, stack_offset);
    } else {
        // Add static offset and store
        auto& aux = aux_regs[0];
        h->addi(aux, ptr_reg, static_cast<int32_t>(ptr_offset));
        h->sw(aux, Xbyak_riscv::sp, stack_offset);
    }
}

}  // namespace ov::intel_cpu::riscv64::utils