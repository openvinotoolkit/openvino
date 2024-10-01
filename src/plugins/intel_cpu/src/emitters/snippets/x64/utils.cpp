// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include "snippets/lowered/expressions/buffer_expression.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/op/loop.hpp"

#include "emitters/utils.hpp"

using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {
namespace utils {

size_t get_buffer_cluster_id(const ov::snippets::lowered::ExpressionPort& port) {
    auto get_cluster_id = [](const snippets::lowered::ExpressionPort& p) {
        const auto buffer = ov::as_type_ptr<ov::snippets::lowered::BufferExpression>(p.get_expr());
        return buffer ? buffer->get_cluster_id() : SIZE_MAX;
    };
    const auto& ma_op = std::dynamic_pointer_cast<ov::snippets::modifier::MemoryAccess>(port.get_expr()->get_node());
    OPENVINO_ASSERT(ma_op, "Expected MemoryAccess op!");
    size_t offset = ov::snippets::utils::get_dynamic_value<size_t>();
    size_t id = SIZE_MAX;
    switch (port.get_type()) {
        case ov::snippets::lowered::ExpressionPort::Type::Input:
            offset = ma_op->get_input_offset(port.get_index());
            id = get_cluster_id(port.get_port_connector_ptr()->get_source());
            break;
        case ov::snippets::lowered::ExpressionPort::Type::Output:
            offset = ma_op->get_output_offset(port.get_index());
            for (const auto& child : port.get_connected_ports())
                if (!ov::is_type<snippets::op::LoopEnd>(child.get_expr()->get_node()))
                    id = get_cluster_id(child);
            break;
        default:
            OV_CPU_JIT_EMITTER_THROW("Uknown type of expression port!");
    }
    OV_CPU_JIT_EMITTER_ASSERT(IMPLICATION(ov::snippets::utils::is_dynamic_value(offset), id != SIZE_MAX),
                              "In dynamic case Buffer Cluster ID must be known!");
    return id;
}

Xbyak::Reg64 get_aux_gpr(const std::vector<size_t>& used_gpr_idxs) {
    // RSP, RBP - stack-related registers, abi_param1 - runtime parameter register in the kernel
    static std::unordered_set<size_t> blacklist_gpr_idxs = { Xbyak::Operand::RSP, Xbyak::Operand::RBP, static_cast<size_t>(abi_param1.getIdx()) };
    for (size_t gpr_idx = 0; gpr_idx <= Xbyak::Operand::R15; ++gpr_idx) {
        size_t _idx = Xbyak::Operand::R15 - gpr_idx; // we allocate from the end
        if (std::find(used_gpr_idxs.cbegin(), used_gpr_idxs.cend(), _idx) != used_gpr_idxs.cend()) continue;
        if (blacklist_gpr_idxs.count(_idx) > 0) continue;
        return Xbyak::Reg64(_idx);
    }
    OV_CPU_JIT_EMITTER_THROW("Failed to allocate aux GPR");
}

void push_ptr_with_runtime_offset_on_stack(dnnl::impl::cpu::x64::jit_generator* h, size_t stack_offset,
                                           Xbyak::Reg64 ptr_reg, Xbyak::Reg64 aux_reg, size_t runtime_offset) {
    const auto stack_frame = h->qword[h->rsp + stack_offset];
    h->mov(aux_reg, ptr_reg);
    h->add(aux_reg, h->ptr[abi_param1 + runtime_offset]);
    h->mov(stack_frame, aux_reg);
}

void push_ptr_with_static_offset_on_stack(dnnl::impl::cpu::x64::jit_generator* h, size_t stack_offset,
                                          Xbyak::Reg64 ptr_reg, size_t ptr_offset) {
    const auto stack_frame = h->qword[h->rsp + stack_offset];
    h->mov(stack_frame, ptr_reg);
    if (ptr_offset != 0) h->add(stack_frame, ptr_offset);
}

}   // namespace utils
}   // namespace intel_cpu
}   // namespace ov
