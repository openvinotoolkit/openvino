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

size_t get_buffer_cluster_id(const ov::snippets::lowered::ExpressionPort& port) {
    auto get_cluster_id = [](const snippets::lowered::ExpressionPort& p) {
        const auto buffer = ov::as_type_ptr<ov::snippets::lowered::BufferExpression>(p.get_expr());
        return buffer ? buffer->get_cluster_id() : SIZE_MAX;
    };
    const auto& ma_op = std::dynamic_pointer_cast<ov::snippets::modifier::MemoryAccess>(port.get_expr()->get_node());
    OPENVINO_ASSERT(ma_op, "Expected MemoryAccess op!");
    auto offset = ov::snippets::utils::get_dynamic_value<size_t>();
    size_t id = SIZE_MAX;
    switch (port.get_type()) {
    case ov::snippets::lowered::ExpressionPort::Type::Input:
        offset = ma_op->get_input_offset(port.get_index());
        id = get_cluster_id(port.get_port_connector_ptr()->get_source());
        break;
    case ov::snippets::lowered::ExpressionPort::Type::Output:
        offset = ma_op->get_output_offset(port.get_index());
        for (const auto& child : port.get_connected_ports()) {
            if (!ov::is_type<snippets::op::LoopEnd>(child.get_expr()->get_node())) {
                id = get_cluster_id(child);
            }
        }
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Uknown type of expression port!");
    }
    OV_CPU_JIT_EMITTER_ASSERT(IMPLICATION(ov::snippets::utils::is_dynamic_value(offset), id != SIZE_MAX),
                              "In dynamic case Buffer Cluster ID must be known!");
    return id;
}

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
    Xbyak_aarch64::XReg offset_reg(4);

    // Handle large runtime offsets by using a temporary register
    if (runtime_offset > 4095) {
        Xbyak_aarch64::XReg temp_offset_reg(6);
        h->mov(temp_offset_reg, static_cast<uint64_t>(runtime_offset));
        h->add(temp_offset_reg, abi_param1, temp_offset_reg);
        h->ldr(offset_reg, Xbyak_aarch64::ptr(temp_offset_reg));
    } else {
        h->ldr(offset_reg, Xbyak_aarch64::ptr(abi_param1, static_cast<int32_t>(runtime_offset)));
    }

    h->add(aux_reg, aux_reg, offset_reg);

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
    Xbyak_aarch64::XReg temp_reg(4);
    h->mov(temp_reg, ptr_reg);

    // For large offsets, use a register to hold the offset value
    if (ptr_offset > 4095) {  // 12-bit immediate limit for add instruction
        Xbyak_aarch64::XReg offset_reg(6);
        h->mov(offset_reg, static_cast<uint64_t>(ptr_offset));
        h->add(temp_reg, temp_reg, offset_reg);
    } else {
        h->add(temp_reg, temp_reg, static_cast<int32_t>(ptr_offset));
    }

    // Store the adjusted pointer on stack
    h->str(temp_reg, Xbyak_aarch64::ptr(h->sp, stack_offset));
}

}  // namespace ov::intel_cpu::aarch64::utils
