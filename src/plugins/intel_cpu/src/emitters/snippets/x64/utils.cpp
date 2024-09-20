// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include "snippets/lowered/expressions/buffer_expression.hpp"
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
    size_t id = SIZE_MAX;
    switch (port.get_type()) {
        case ov::snippets::lowered::ExpressionPort::Type::Input:
            id = get_cluster_id(port.get_port_connector_ptr()->get_source());
            break;
        case ov::snippets::lowered::ExpressionPort::Type::Output:
            for (const auto& child : port.get_connected_ports())
                if (!ov::is_type<snippets::op::LoopEnd>(child.get_expr()->get_node()))
                    id = get_cluster_id(child);
            break;
        default:
            OV_CPU_JIT_EMITTER_THROW("Uknown type of expression port!");
    }
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

void write_data_ptr_on_stack(dnnl::impl::cpu::x64::jit_generator* h, size_t stack_offset, Xbyak::Reg64 ptr_reg, Xbyak::Reg64 aux_reg,
                             size_t ptr_offset, size_t runtime_offset) {
    const auto stack_frame = h->qword[h->rsp + stack_offset];
    h->mov(aux_reg, ptr_reg);
    if (snippets::utils::is_dynamic_value(ptr_offset))
        h->add(aux_reg,  h->ptr[abi_param1 + runtime_offset]);
    else if (ptr_offset != 0)
        h->add(aux_reg, ptr_offset);
    h->mov(stack_frame, aux_reg);
}

}   // namespace utils
}   // namespace intel_cpu
}   // namespace ov
