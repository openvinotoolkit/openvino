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

Xbyak_aarch64::XReg init_memory_access_aux_gpr(const std::vector<size_t>& used_gpr_reg_idxs,
                                               const std::vector<size_t>& aux_gpr_idxs,
                                               std::set<snippets::Reg>& regs_to_spill) {
    if (!aux_gpr_idxs.empty()) {
        return Xbyak_aarch64::XReg(static_cast<int>(aux_gpr_idxs[0]));
    }
    const auto aux_reg = ov::intel_cpu::aarch64::utils::get_aux_gpr(used_gpr_reg_idxs);
    regs_to_spill.emplace(snippets::RegType::gpr, aux_reg.getIdx());
    return aux_reg;
}

void push_ptr_with_runtime_offset_on_stack(dnnl::impl::cpu::aarch64::jit_generator* h,
                                           int32_t stack_offset,
                                           const Xbyak_aarch64::XReg& ptr_reg,
                                           const Xbyak_aarch64::XReg& aux_reg,
                                           size_t runtime_offset) {
    // Copy pointer to aux register
    h->mov(aux_reg, ptr_reg);

    // Load the runtime offset from abi_param1 (X0) and add it to the pointer
    Xbyak_aarch64::XReg abi_param1(0);
    Xbyak_aarch64::XReg temp_reg(h->X_TMP_0);

    // Load the offset value from the runtime parameter location
    h->add_imm(temp_reg, abi_param1, runtime_offset, Xbyak_aarch64::XReg(h->X_TMP_1));
    h->ldr(temp_reg, Xbyak_aarch64::ptr(temp_reg));

    h->add(aux_reg, aux_reg, temp_reg);

    // Store the adjusted pointer on stack
    h->str(aux_reg, Xbyak_aarch64::ptr(h->sp, stack_offset));
}

void push_ptr_with_static_offset_on_stack(dnnl::impl::cpu::aarch64::jit_generator* h,
                                          int32_t stack_offset,
                                          const Xbyak_aarch64::XReg& ptr_reg,
                                          size_t ptr_offset) {
    // If there's no static offset, just store the pointer
    if (ptr_offset == 0) {
        h->str(ptr_reg, Xbyak_aarch64::ptr(h->sp, stack_offset));
        return;
    }

    // For non-zero offsets, apply the offset and then store
    Xbyak_aarch64::XReg temp_reg(h->X_TMP_0);
    h->add_imm(temp_reg, ptr_reg, ptr_offset, Xbyak_aarch64::XReg(h->X_TMP_1));

    // Store the adjusted pointer on stack
    h->str(temp_reg, Xbyak_aarch64::ptr(h->sp, stack_offset));
}

}  // namespace ov::intel_cpu::aarch64::utils
