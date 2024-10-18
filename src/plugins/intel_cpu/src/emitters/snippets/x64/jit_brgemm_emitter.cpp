// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_brgemm_emitter.hpp"

#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "snippets/utils/utils.hpp"
#include "emitters/plugin/x64/utils.hpp"
#include "utils.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"

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
    const auto brgemm_type = brgemm_node->get_type();
    BrgemmKernelConfig kernel_config(brg0Prc, brg1Prc, with_amx(brgemm_type), with_compensations(brgemm_type),
                                     brgemm_utils::get_primitive_isa(brg0Prc, with_amx(brgemm_type)));
    m_kernel_executor = kernel_table->register_kernel<BrgemmKernelExecutor>(expr,
                                                                            compiled_kernel_cache,
                                                                            kernel_config);
    // Note: even if the Brgemm node is dynamic, the first shapeInfer and RuntimeConfigurator::update()
    // are performed before the BrgemmKernelExecutor registration. So we have to trigger update() manually
    // for both static and the 1st dynamic shapes.
    OV_CPU_JIT_EMITTER_ASSERT(!snippets::utils::is_dynamic_vdims(expr->get_input_port_descriptor(0)->get_shape()) &&
                              !snippets::utils::is_dynamic_vdims(expr->get_input_port_descriptor(1)->get_shape()),
                              "Jit emitter is called when the shapes are unknown");

    m_memory_offsets = {brgemm_node->get_offset_a(), brgemm_node->get_offset_b(), brgemm_node->get_offset_c()};
    m_buffer_ids = {utils::get_buffer_cluster_id(expr->get_input_port(0)), utils::get_buffer_cluster_id(expr->get_input_port(1)),
                    utils::get_buffer_cluster_id(expr->get_output_port(0))};
    if (with_scratchpad(brgemm_type)) {
        m_memory_offsets.push_back(brgemm_node->get_offset_scratch());
        m_buffer_ids.push_back(utils::get_buffer_cluster_id(expr->get_input_port(2)));
    }
}

std::set<std::vector<element::Type>> jit_brgemm_emitter::get_supported_precisions(const std::shared_ptr<ov::Node>& node) {
    const auto brgemm = as_type_ptr<ov::intel_cpu::BrgemmCPU>(node);
    OV_CPU_JIT_EMITTER_ASSERT(brgemm, "get_supported_precisions() expects BrgemmCPU node");
    using brgemm_utils::BRGEMM_TYPE;
    if (brgemm->get_type() == BRGEMM_TYPE::STAND_ALONE) {
        return {{element::f32, element::f32}};
    } else if (brgemm->get_type() == BRGEMM_TYPE::REPACKING_ONLY) {
        std::set<std::vector<element::Type>> supported_types = {{element::u8, element::i8},
                                                                {element::bf16, element::bf16},
                                                                {element::f32, element::f32}};
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni_2))
            supported_types.insert({element::i8, element::i8});
        return supported_types;
    } else if (brgemm->get_type() == BRGEMM_TYPE::WITH_COMPENSATIONS) {
        return {{element::i8, element::i8, element::f32}};
    } else if (brgemm->get_type() == BRGEMM_TYPE::WITH_AMX) {
        return {{element::i8, element::i8, element::u8},
                {element::u8, element::i8, element::u8},
                {element::bf16, element::bf16, element::u8}};
    }
    OV_CPU_JIT_EMITTER_THROW("got BrgemmCPU node with unsupported type");
}

void jit_brgemm_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OV_CPU_JIT_EMITTER_ASSERT(m_memory_offsets.size() == in.size() + 1 && (out.size() == 1),
                              "expects 3 inputs if there are compensations/wsp");
}

void jit_brgemm_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    std::vector<size_t> mem_ptrs_idxs{in[0], in[1], out[0]};
    if (in.size() > 2)
        mem_ptrs_idxs.emplace_back(in[2]);

    EmitABIRegSpills spill(h);
    spill.preamble();

    h->mov(h->rbp, reinterpret_cast<uint64_t>(BrgemmKernelExecutor::execute));
    auto reserved_stack_size = sizeof(BrgemmKernelExecutor::call_args);
    // Reserve memory on the stack
    h->sub(h->rsp, reserved_stack_size);

    const bool is_dynamic_case = std::any_of(m_memory_offsets.cbegin(), m_memory_offsets.cend(), ov::snippets::utils::is_dynamic_value<size_t>);
    Xbyak::Reg64 aux_reg = is_dynamic_case ? ov::intel_cpu::utils::get_aux_gpr(mem_ptrs_idxs) : Xbyak::Reg64();

    const std::vector<size_t> brgemm_args_offsets {GET_OFF_BRGEMM_ARGS(A), GET_OFF_BRGEMM_ARGS(B), GET_OFF_BRGEMM_ARGS(C), GET_OFF_BRGEMM_ARGS(scratch)};
    const auto& mem_ptrs = utils::transform_idxs_to_regs(mem_ptrs_idxs);
    for (size_t i = 0; i < mem_ptrs.size(); i++) {
        if (ov::snippets::utils::is_dynamic_value(m_memory_offsets[i]))
            utils::push_ptr_with_runtime_offset_on_stack(h, brgemm_args_offsets[i], mem_ptrs[i], aux_reg,
                                                         GET_OFF(buffer_offsets) + m_buffer_ids[i] * sizeof(size_t));
        else
            utils::push_ptr_with_static_offset_on_stack(h, brgemm_args_offsets[i], mem_ptrs[i], m_memory_offsets[i]);
    }

    // No scratchpad => need to write nullptr manually
    if (mem_ptrs.size() < 4)
        h->mov(h->qword[h->rsp + brgemm_args_offsets.back()], reinterpret_cast<uintptr_t>(nullptr));

    // abi_param1 always contains jit_snippets_call_args which has amx tile config for each thread
    h->lea(h->r10, h->ptr[abi_param1 + GET_OFF(amx_tile_config)]);
    h->mov(h->qword[h->rsp + GET_OFF_BRGEMM_ARGS(amx_tile_config)], h->r10);

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
