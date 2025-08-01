// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <xbyak/xbyak.h>

#include <algorithm>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <set>
#include <unordered_set>
#include <vector>

#include "emitters/utils.hpp"
#include "snippets/emitter.hpp"

using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu::utils {

Xbyak::Reg64 get_aux_gpr(const std::vector<size_t>& used_gpr_idxs) {
    // RSP - stack pointer should be preserved, abi_param1 and abi_param2 - runtime parameter register in the kernel
    static std::unordered_set<size_t> blacklist_gpr_idxs = {Xbyak::Operand::RSP,
                                                            static_cast<size_t>(abi_param1.getIdx()),
                                                            static_cast<size_t>(abi_param2.getIdx())};
    for (size_t gpr_idx = 0; gpr_idx <= Xbyak::Operand::R15; ++gpr_idx) {
        size_t _idx = Xbyak::Operand::R15 - gpr_idx;  // we allocate from the end
        if (std::find(used_gpr_idxs.cbegin(), used_gpr_idxs.cend(), _idx) != used_gpr_idxs.cend()) {
            continue;
        }
        if (blacklist_gpr_idxs.count(_idx) > 0) {
            continue;
        }
        return Xbyak::Reg64(_idx);
    }
    OV_CPU_JIT_EMITTER_THROW("Failed to allocate aux GPR");
}

Xbyak::Reg64 init_memory_access_aux_gpr(const std::vector<size_t>& used_gpr_reg_idxs,
                                        const std::vector<size_t>& aux_gpr_idxs,
                                        std::set<snippets::Reg>& regs_to_spill) {
    if (!aux_gpr_idxs.empty()) {
        return Xbyak::Reg64(static_cast<int>(aux_gpr_idxs[0]));
    }
    const auto aux_reg = ov::intel_cpu::utils::get_aux_gpr(used_gpr_reg_idxs);
    regs_to_spill.emplace(snippets::RegType::gpr, aux_reg.getIdx());
    return aux_reg;
}

void push_ptr_with_runtime_offset_on_stack(dnnl::impl::cpu::x64::jit_generator_t* h,
                                           size_t stack_offset,
                                           Xbyak::Reg64 ptr_reg,
                                           Xbyak::Reg64 aux_reg,
                                           size_t runtime_offset) {
    const auto stack_frame = h->qword[h->rsp + stack_offset];
    h->mov(aux_reg, ptr_reg);
    h->add(aux_reg, h->ptr[abi_param1 + runtime_offset]);
    h->mov(stack_frame, aux_reg);
}

void push_ptr_with_static_offset_on_stack(dnnl::impl::cpu::x64::jit_generator_t* h,
                                          size_t stack_offset,
                                          Xbyak::Reg64 ptr_reg,
                                          size_t ptr_offset) {
    const auto stack_frame = h->qword[h->rsp + stack_offset];
    h->mov(stack_frame, ptr_reg);  // move to value in address
    if (ptr_offset != 0) {
        h->add(stack_frame, ptr_offset);   // the value in address add offset
    }
}

}  // namespace ov::intel_cpu::utils
