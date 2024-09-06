// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_brgemm_copy_b_emitter.hpp"

#include "emitters/plugin/x64/utils.hpp"
#include "emitters/snippets/x64/utils.hpp"

#include "snippets/utils/utils.hpp"
#include "snippets/lowered/expression.hpp"

#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/matmul/brgemm_matmul_utils.hpp>


using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace ov::intel_cpu::brgemm_utils;
using namespace ov::snippets::utils;

namespace ov {
namespace intel_cpu {
namespace {
bool get_is_transposed(const ov::snippets::lowered::ExpressionPtr& expr) {
    const auto& layout = expr->get_input_port_descriptor(0)->get_layout();
    const auto is_transposed = !layout.empty() && layout.back() != layout.size() - 1;
    OV_CPU_JIT_EMITTER_ASSERT(IMPLICATION(is_transposed, (layout[layout.size() - 2] == layout.size() - 1)),
                              "supports only N dim placed as last or pre last dimension");
    return is_transposed;
}
}  // namespace


jit_brgemm_copy_b_emitter::jit_brgemm_copy_b_emitter(jit_generator* h, cpu_isa_t isa, const ov::snippets::lowered::ExpressionPtr& expr,
                                                     const snippets::KernelExecutorTablePtr& kernel_table,
                                                     const ov::intel_cpu::MultiCacheWeakPtr& compiled_kernel_cache)
    : jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto brgemm_repack = ov::as_type_ptr<ov::intel_cpu::BrgemmCopyB>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_repack, "expects BrgemmCopyB node");

    // Note: even if the BrgemmCopyB node is dynamic, the first shapeInfer and RuntimeConfigurator::update()
    // are performed before the BrgemmCopyBKernelExecutor registration. So we have to trigger update() manually
    // for both static and the 1st dynamic shapes.
    OV_CPU_JIT_EMITTER_ASSERT(!snippets::utils::is_dynamic_vdims(expr->get_input_port_descriptor(0)->get_shape()),
                              "Jit emitter is called when the shapes are unknown");

    const auto is_transposed = get_is_transposed(expr);
    // However, the generated primitive can be also applied to tensors with other ranks
    const auto format = is_transposed ? dnnl_ba : dnnl_ab;

    const auto& in_subtensor = get_projected_subtensor(expr->get_input_port(0));
    const auto N_blk = *in_subtensor.rbegin();
    const auto K_blk = *++in_subtensor.rbegin();

    const auto& src_prc = brgemm_repack->get_src_element_type();
    const auto& wei_prc = brgemm_repack->get_input_element_type(0);
    const auto wei_N_blk = brgemm_utils::repacking::compute_inner_n_block(wei_prc);
    const auto brgemm_type = get_brgemm_type(src_prc, K_blk, N_blk, is_transposed);
    const auto primitive_isa = brgemm_utils::get_primitive_isa(src_prc, with_amx(brgemm_type));
    m_with_comp = with_compensations(brgemm_type);

    BrgemmCopyBKernelConfig kernel_config(src_prc, wei_prc, format, primitive_isa, m_with_comp, is_transposed, wei_N_blk);
    m_kernel_executor = kernel_table->register_kernel<BrgemmCopyBKernelExecutor>(expr, compiled_kernel_cache, kernel_config);

    m_memory_offsets = {brgemm_repack->get_offset_in(), brgemm_repack->get_offset_out()};
    m_buffer_ids = {utils::get_buffer_cluster_id(expr->get_input_port(0), m_memory_offsets[0]),
                    utils::get_buffer_cluster_id(expr->get_output_port(0), m_memory_offsets[1])};
    if (m_with_comp) {
        m_memory_offsets.push_back(brgemm_repack->get_offset_compensations());
        m_buffer_ids.push_back(utils::get_buffer_cluster_id(expr->get_output_port(1), m_memory_offsets.back()));
    }
}

void jit_brgemm_copy_b_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == 1, "expects 1 input");
    OV_CPU_JIT_EMITTER_ASSERT((m_with_comp && out.size() == 2) || (!m_with_comp && out.size() == 1),
                              "expects 2 outputs if there are compensations");
}

void jit_brgemm_copy_b_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    std::vector<size_t> mem_ptrs_idxs{in[0], out[0]};
    if (out.size() > 1)
        mem_ptrs_idxs.emplace_back(out[1]);
    emit_kernel_call(mem_ptrs_idxs, m_memory_offsets);
}

void jit_brgemm_copy_b_emitter::emit_kernel_call(const std::vector<size_t>& mem_ptrs_idxs, const std::vector<size_t>& mem_offsets) const {
    JitSafeInternalCall safe_internal_caller(h, host_isa_);

    h->mov(h->rbp, reinterpret_cast<uint64_t>(BrgemmCopyBKernelExecutor::execute));
    auto reserved_stack_size = sizeof(BrgemmCopyBKernel::call_args);
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
    const std::vector<size_t> args_offsets {GET_OFF_BRGEMM_COPY_B_ARGS(src), GET_OFF_BRGEMM_COPY_B_ARGS(tr_src), GET_OFF_BRGEMM_COPY_B_ARGS(compensation_ptr)};
    const auto& mem_ptrs = ov::intel_cpu::utils::transform_idxs_to_regs(mem_ptrs_idxs);
    for (size_t i = 0; i < mem_ptrs.size(); i++)
        write_addr_on_stack(args_offsets[i], mem_ptrs[i], mem_offsets[i], m_buffer_ids[i]);

    // No scratchpad => need to write nullptr manually
    if (!m_with_comp)
        h->mov(h->qword[h->rsp + args_offsets.back()], reinterpret_cast<uintptr_t>(nullptr));

    h->mov(abi_param1, reinterpret_cast<uintptr_t>(m_kernel_executor.get()));
    h->mov(abi_param2, h->rsp);

    safe_internal_caller.call(h->rbp);

    h->add(h->rsp, reserved_stack_size);
}

}   // namespace intel_cpu
}   // namespace ov
