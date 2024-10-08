// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_brgemm_copy_a_emitter.hpp"

#include "emitters/plugin/x64/utils.hpp"
#include "emitters/snippets/x64/utils.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"

#include "snippets/utils/utils.hpp"

#include "transformations/snippets/x64/op/brgemm_copy_a.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"


using namespace dnnl::impl::cpu::x64;
using namespace ov::intel_cpu::brgemm_utils;
using namespace ov::snippets::utils;

namespace ov {
namespace intel_cpu {

jit_brgemm_copy_a_emitter::jit_brgemm_copy_a_emitter(jit_generator* h, cpu_isa_t isa, const ov::snippets::lowered::ExpressionPtr& expr,
                                                     const snippets::KernelExecutorTablePtr& kernel_table,
                                                     const ov::intel_cpu::MultiCacheWeakPtr& compiled_kernel_cache)
    : jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto brgemm_repack = ov::as_type_ptr<ov::intel_cpu::BrgemmCopyA>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(brgemm_repack, "expects BrgemmCopyA node");

    // Note: even if the BrgemmCopyA node is dynamic, the first shapeInfer and RuntimeConfigurator::update()
    // are performed before the BrgemmCopyAKernelExecutor registration. So we have to trigger update() manually
    // for both static and the 1st dynamic shapes.
    OV_CPU_JIT_EMITTER_ASSERT(!snippets::utils::is_dynamic_vdims(expr->get_input_port_descriptor(0)->get_shape()),
                              "Jit emitter is called when the shapes are unknown");

    const auto& brgemm_config = brgemm_repack->get_config();
    BrgemmCopyAKernelConfig kernel_config(brgemm_repack->get_input_element_type(0), brgemm_config.isa());
    m_kernel_executor = kernel_table->register_kernel<BrgemmCopyAKernelExecutor>(expr, compiled_kernel_cache, kernel_config);

    m_memory_offsets = {brgemm_repack->get_offset_in(), brgemm_repack->get_offset_out()};
    m_buffer_ids = {utils::get_buffer_cluster_id(expr->get_input_port(0)), utils::get_buffer_cluster_id(expr->get_output_port(0))};
}

void jit_brgemm_copy_a_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == 1 && out.size() == 1, "expects 1 input and 1 output");
}

void jit_brgemm_copy_a_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);

    std::vector<size_t> mem_ptrs_idxs{in[0], out[0]};

    EmitABIRegSpills spill(h);
    spill.preamble();

    h->mov(h->rbp, reinterpret_cast<uint64_t>(BrgemmCopyAKernelExecutor::execute));
    auto reserved_stack_size = sizeof(BrgemmCopyAKernel::call_args);
    // Reserve memory on the stack
    h->sub(h->rsp, reserved_stack_size);

    const bool is_dynamic_case = std::any_of(m_memory_offsets.cbegin(), m_memory_offsets.cend(), ov::snippets::utils::is_dynamic_value<size_t>);
    Xbyak::Reg64 aux_reg = is_dynamic_case ? ov::intel_cpu::utils::get_aux_gpr(mem_ptrs_idxs) : Xbyak::Reg64();

    const std::vector<size_t> args_offsets {GET_OFF_BRGEMM_COPY_A_ARGS(src), GET_OFF_BRGEMM_COPY_A_ARGS(tr_src)};
    const auto& mem_ptrs = ov::intel_cpu::utils::transform_idxs_to_regs(mem_ptrs_idxs);
    for (size_t i = 0; i < mem_ptrs.size(); i++) {
        if (ov::snippets::utils::is_dynamic_value(m_memory_offsets[i]))
            utils::push_ptr_with_runtime_offset_on_stack(h, args_offsets[i], mem_ptrs[i], aux_reg,
                                                         GET_OFF(buffer_offsets) + m_buffer_ids[i] * sizeof(size_t));
        else
            utils::push_ptr_with_static_offset_on_stack(h, args_offsets[i], mem_ptrs[i], m_memory_offsets[i]);
    }

    h->mov(abi_param1, reinterpret_cast<uintptr_t>(m_kernel_executor.get()));
    h->mov(abi_param2, h->rsp);

    spill.rsp_align();
    h->call(h->rbp);
    spill.rsp_restore();

    h->add(h->rsp, reserved_stack_size);

    spill.postamble();
}

}   // namespace intel_cpu
}   // namespace ov
