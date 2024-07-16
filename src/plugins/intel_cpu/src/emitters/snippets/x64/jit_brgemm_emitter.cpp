// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_brgemm_emitter.hpp"

#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/amx_tile_configure.hpp>
#include "snippets/utils.hpp"
#include "utils.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

jit_brgemm_emitter::jit_brgemm_emitter(jit_generator* h, cpu_isa_t isa,
                                       const ov::snippets::lowered::ExpressionPtr& expr,
                                       const snippets::KernelExecutorTablePtr& kernel_table,
                                       const ov::intel_cpu::MultiCacheWeakPtr& compiled_kernel_cache) :
                                       jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto& brgemm_node = as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node());
    const auto& brg0Prc = brgemm_node->get_input_element_type(0);
    const auto& brg1Prc = brgemm_node->get_input_element_type(1);
    BrgemmKernelConfig kernel_config(brg0Prc, brg1Prc,
                                      brgemm_node->get_beta(), brgemm_node->is_amx(),
                                      brgemm_node->is_with_compensations());
    m_kernel_executor = kernel_table->register_kernel<BrgemmKernelExecutor>(expr,
                                                                            compiled_kernel_cache,
                                                                            kernel_config);
    // Note: even if the Brgemm node is dynamic, the first shapeInfer and RuntimeConfigurator::update()
    // are performed before the BrgemmKernelExecutor registration. So we have to trigger update() manually
    // for both static and the 1st dynamic shapes.
    OV_CPU_JIT_EMITTER_ASSERT(!snippets::utils::is_dynamic_vdims(expr->get_input_port_descriptor(0)->get_shape()) &&
                              !snippets::utils::is_dynamic_vdims(expr->get_input_port_descriptor(1)->get_shape()),
                              "Jit emitter is called when the shapes are unknown");
    auto get_cluster_id = [](const snippets::lowered::ExpressionPort& p) {
        // Note: NewMemoryBuffer is used as a scratchpad and can't be dynamic, so we don't need to account for them here
        if (const auto buffer = ov::as_type_ptr<ov::snippets::op::IntermediateMemoryBuffer>(p.get_expr()->get_node()))
            return buffer->get_cluster_id();
        else
            return SIZE_MAX;
    };
    m_memory_offsets = {brgemm_node->get_offset_a(), brgemm_node->get_offset_b(), brgemm_node->get_offset_c()};
    if (brgemm_node->is_with_scratchpad())
        m_memory_offsets.push_back(brgemm_node->get_offset_scratch());

    m_buffer_ids.assign(m_memory_offsets.size(), SIZE_MAX);
    for (size_t i = 0; i < m_memory_offsets.size(); i++) {
         if (snippets::utils::is_dynamic_value(m_memory_offsets[i])) {
             switch (i) {
                 case 0:
                 case 1:
                     m_buffer_ids[i] = get_cluster_id(expr->get_input_port_connector(i)->get_source());
                     break;
                 case 2:
                     for (const auto& child : expr->get_output_port_connector(0)->get_consumers())
                         if (!ov::is_type<snippets::op::LoopEnd>(child.get_expr()->get_node()))
                             m_buffer_ids[i] = get_cluster_id(child);
             }
             OV_CPU_JIT_EMITTER_ASSERT(m_buffer_ids[i] != SIZE_MAX, "Dynamic offset requires a valid buffer ID");
         }
    }
}

std::set<std::vector<element::Type>> jit_brgemm_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    const auto brgemm = as_type_ptr<ov::intel_cpu::BrgemmCPU>(node);
    OV_CPU_JIT_EMITTER_ASSERT(brgemm, "get_supported_precisions() expects BrgemmCPU node");
    switch (brgemm->get_type()) {
        case BrgemmCPU::Type::Floating:
            return {{element::f32, element::f32}};
        case BrgemmCPU::Type::WithDataRepacking:
            return {{element::u8, element::i8},
                    {element::bf16, element::bf16}};
        case BrgemmCPU::Type::WithCompensations:
            return {{element::i8, element::i8, element::f32}};
        case BrgemmCPU::Type::AMX:
            return {{element::i8, element::i8, element::u8},
                    {element::u8, element::i8, element::u8},
                    {element::bf16, element::bf16, element::u8}};
        default:
            OV_CPU_JIT_EMITTER_THROW("got BrgemmCPU node with unsupported type");
    }
}

void jit_brgemm_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(m_memory_offsets.size() == in.size() + 1 && (out.size() == 1),
                              "expects 3 inputs if there are compensations/wsp");
}

void jit_brgemm_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    if (host_isa_ == cpu::x64::avx512_core) {
        std::vector<size_t> mem_ptrs_idxs{in[0], in[1], out[0]};
        if (in.size() > 2)
            mem_ptrs_idxs.emplace_back(in[2]);
        emit_brgemm_kernel_call(mem_ptrs_idxs, m_memory_offsets);
    } else {
        OV_CPU_JIT_EMITTER_THROW("requires at least avx512_core instruction set");
    }
}
void jit_brgemm_emitter::emit_brgemm_kernel_call(const std::vector<size_t>& mem_ptrs_idxs, const std::vector<size_t>& mem_offsets) const {
    internal_call_preamble();
    h->mov(h->rbp, reinterpret_cast<uint64_t>(BrgemmKernelExecutor::execute));
    auto reserved_stack_size = sizeof(BrgemmKernelExecutor::call_args);
    // Reserve memory on the stack
    h->sub(h->rsp, reserved_stack_size);

    Xbyak::Reg64 aux_reg = [this, &mem_ptrs_idxs]() {
        std::set<size_t> used(mem_ptrs_idxs.begin(), mem_ptrs_idxs.end());
        std::vector<Xbyak::Reg64> spilled_gprs {h->r8, h->r9, h->r10, h->r11, h->r12, h->r13, h->r14, h->r15,
                                                h->rax, h->rbx, h->rcx, h->rdx, h->rdi, h->rsi, h->rbp};
        for (const auto& reg : spilled_gprs)
            if (used.count(reg.getIdx()) == 0)
                return reg;
        OV_CPU_JIT_EMITTER_THROW("Failed to allocate aux register");
    }();

    auto write_addr_on_stack = [&](size_t arg_offset, Reg64 addr, size_t addr_offset, size_t buffer_id) {
        const auto stack_frame = h->qword[h->rsp + arg_offset];
        h->mov(aux_reg, addr);
        if (snippets::utils::is_dynamic_value(addr_offset))
            h->add(aux_reg,  h->ptr[abi_param1 + GET_OFF(buffer_offsets) + buffer_id * sizeof(size_t)]);
        else if (addr_offset != 0)
            h->add(aux_reg, addr_offset);
        h->mov(stack_frame, aux_reg);
    };
    const std::vector<size_t> brgemm_args_offsets {GET_OFF_BRGEMM_ARGS(A), GET_OFF_BRGEMM_ARGS(B), GET_OFF_BRGEMM_ARGS(C),
                                                   GET_OFF_BRGEMM_ARGS(scratch)};
    const auto& mem_ptrs = utils::transform_idxs_to_regs(mem_ptrs_idxs);
    for (size_t i = 0; i < mem_ptrs.size(); i++)
        write_addr_on_stack(brgemm_args_offsets[i], mem_ptrs[i], mem_offsets[i], m_buffer_ids[i]);

    // No scratchpad => need to write nullptr manually
    if (mem_ptrs.size() < 4)
        h->mov(h->qword[h->rsp + brgemm_args_offsets.back()], reinterpret_cast<uintptr_t>(nullptr));

    // abi_param1 always contains jit_snippets_call_args which has amx tile config for each thread
    h->lea(h->r10, h->ptr[abi_param1 + GET_OFF(amx_tile_config)]);
    h->mov(h->qword[h->rsp + GET_OFF_BRGEMM_ARGS(amx_tile_config)], h->r10);

    h->mov(abi_param1, reinterpret_cast<uintptr_t>(m_kernel_executor.get()));
    h->mov(abi_param2, h->rsp);

    internal_call_rsp_align();
    h->call(h->rbp);
    internal_call_rsp_restore();

    h->add(h->rsp, reserved_stack_size);
    internal_call_postamble();
}

}   // namespace intel_cpu
}   // namespace ov
