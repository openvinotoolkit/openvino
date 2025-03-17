// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_brgemm_emitter.hpp"

#include "emitters/plugin/x64/utils.hpp"
#include "emitters/snippets/x64/kernel_executors/brgemm.hpp"
#include "emitters/snippets/x64/kernel_executors/brgemm_amx.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"
#include "utils.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace ov::intel_cpu::x64;

namespace ov::intel_cpu {

jit_brgemm_emitter::jit_brgemm_emitter(jit_generator* h,
                                       cpu_isa_t isa,
                                       const ov::snippets::lowered::ExpressionPtr& expr,
                                       const snippets::KernelExecutorTablePtr& kernel_table,
                                       const ov::intel_cpu::MultiCacheWeakPtr& compiled_kernel_cache)
    : jit_binary_call_emitter(h, isa, expr->get_live_regs()) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto& brgemm_node = as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node());
    const auto& brg0Prc = brgemm_node->get_input_element_type(0);
    const auto& brg1Prc = brgemm_node->get_input_element_type(1);
    const auto brgemm_type = brgemm_node->get_type();
    m_is_with_amx = brgemm_utils::with_amx(brgemm_type);
    if (m_is_with_amx) {
        BrgemmAMXKernelConfig kernel_config(brg0Prc, brg1Prc, brgemm_utils::get_primitive_isa(brg0Prc, true));
        m_kernel_executor =
            kernel_table->register_kernel<BrgemmAMXKernelExecutor>(expr, compiled_kernel_cache, kernel_config);
    } else {
        BrgemmKernelConfig kernel_config(brg0Prc,
                                         brg1Prc,
                                         with_compensations(brgemm_type),
                                         brgemm_utils::get_primitive_isa(brg0Prc, false));
        m_kernel_executor =
            kernel_table->register_kernel<BrgemmKernelExecutor>(expr, compiled_kernel_cache, kernel_config);
    }
    // Note: even if the Brgemm node is dynamic, the first shapeInfer and RuntimeConfigurator::update()
    // are performed before the BrgemmKernelExecutor registration. So we have to trigger update() manually
    // for both static and the 1st dynamic shapes.
    OV_CPU_JIT_EMITTER_ASSERT(!snippets::utils::is_dynamic_vdims(expr->get_input_port_descriptor(0)->get_shape()) &&
                                  !snippets::utils::is_dynamic_vdims(expr->get_input_port_descriptor(1)->get_shape()),
                              "Jit emitter is called when the shapes are unknown");

    m_memory_offsets = {brgemm_node->get_offset_a(), brgemm_node->get_offset_b(), brgemm_node->get_offset_c()};
    m_buffer_ids = {utils::get_buffer_cluster_id(expr->get_input_port(0)),
                    utils::get_buffer_cluster_id(expr->get_input_port(1)),
                    utils::get_buffer_cluster_id(expr->get_output_port(0))};
    if (with_scratchpad(brgemm_type)) {
        m_memory_offsets.push_back(brgemm_node->get_offset_scratch());
        m_buffer_ids.push_back(utils::get_buffer_cluster_id(expr->get_input_port(2)));
    }
}

std::set<std::vector<element::Type>> jit_brgemm_emitter::get_supported_precisions(
    const std::shared_ptr<ov::Node>& node) {
    const auto brgemm = as_type_ptr<ov::intel_cpu::BrgemmCPU>(node);
    OV_CPU_JIT_EMITTER_ASSERT(brgemm, "get_supported_precisions() expects BrgemmCPU node");
    using brgemm_utils::BRGEMM_TYPE;
    switch (brgemm->get_type()) {
    case BRGEMM_TYPE::STAND_ALONE:
        return {{element::f32, element::f32}};
    case BRGEMM_TYPE::REPACKING_ONLY: {
        std::set<std::vector<element::Type>> supported_types = {{element::u8, element::i8},
                                                                {element::bf16, element::bf16},
                                                                {element::f32, element::f32}};
        if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2_vnni_2)) {
            supported_types.insert({element::i8, element::i8});
        }
        return supported_types;
    }
    case BRGEMM_TYPE::WITH_COMPENSATIONS:
        return {{element::i8, element::i8, element::f32}};
    case BRGEMM_TYPE::WITH_AMX:
        return {{element::i8, element::i8, element::u8},
                {element::u8, element::i8, element::u8},
                {element::bf16, element::bf16, element::u8},
                {element::f16, element::f16, element::u8}};
    default:
        OV_CPU_JIT_EMITTER_THROW("got BrgemmCPU node with unsupported type");
    }
}

void jit_brgemm_emitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(m_memory_offsets.size() == in.size() + 1 && (out.size() == 1),
                              "expects 3 inputs if there are compensations/wsp");
}

void jit_brgemm_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    std::vector<size_t> mem_ptrs_idxs{in[0], in[1], out[0]};
    init_binary_call_regs(2, mem_ptrs_idxs);
    if (in.size() > 2) {
        mem_ptrs_idxs.emplace_back(in[2]);
    }

    if (std::dynamic_pointer_cast<BrgemmAMXKernelExecutor>(m_kernel_executor)) {
        emit_call<BrgemmAMXKernelExecutor>(mem_ptrs_idxs);
    } else if (std::dynamic_pointer_cast<BrgemmKernelExecutor>(m_kernel_executor)) {
        emit_call<BrgemmKernelExecutor>(mem_ptrs_idxs);
    } else {
        OV_CPU_JIT_EMITTER_THROW("uknown execuor type");
    }
}

template <typename T, std::enable_if_t<std::is_base_of_v<BrgemmBaseKernelExecutor, T>, bool>>
void jit_brgemm_emitter::emit_call(const std::vector<size_t>& mem_ptrs_idxs) const {
    const Xbyak::Reg64& aux_reg = get_call_address_reg();
    const Xbyak::Reg64& callee_saved_reg = get_callee_saved_reg();

    EmitABIRegSpills spill(h);
    spill.preamble(get_regs_to_spill());

    auto reserved_stack_size = sizeof(typename T::call_args);
    // Reserve memory on the stack
    h->sub(h->rsp, reserved_stack_size);

#define GET_OFF_CALL_ARGS(field) offsetof(typename T::call_args, field)
    const std::vector<size_t> brgemm_args_offsets = {GET_OFF_CALL_ARGS(A),
                                                     GET_OFF_CALL_ARGS(B),
                                                     GET_OFF_CALL_ARGS(C),
                                                     GET_OFF_CALL_ARGS(scratch)};
#undef GET_OFF_CALL_ARGS

    const auto& mem_ptrs = utils::transform_idxs_to_regs(mem_ptrs_idxs);
    for (size_t i = 0; i < mem_ptrs.size(); i++) {
        if (ov::snippets::utils::is_dynamic_value(m_memory_offsets[i])) {
            utils::push_ptr_with_runtime_offset_on_stack(h,
                                                         brgemm_args_offsets[i],
                                                         mem_ptrs[i],
                                                         aux_reg,
                                                         GET_OFF(buffer_offsets) + m_buffer_ids[i] * sizeof(size_t));
        } else {
            utils::push_ptr_with_static_offset_on_stack(h, brgemm_args_offsets[i], mem_ptrs[i], m_memory_offsets[i]);
        }
    }

    // No scratchpad => need to write nullptr manually
    if (mem_ptrs.size() < 4) {
        h->mov(h->qword[h->rsp + brgemm_args_offsets.back()], reinterpret_cast<uintptr_t>(nullptr));
    }

    // abi_param1 always contains jit_snippets_call_args which has amx tile config for each thread
    if (std::is_same<T, BrgemmAMXKernelExecutor>()) {
        h->lea(aux_reg, h->ptr[abi_param1 + GET_OFF(amx_tile_config)]);
        h->mov(h->qword[h->rsp + GET_OFF_BRGEMM_AMX_ARGS(amx_tile_config)], aux_reg);
    }
    h->mov(aux_reg, reinterpret_cast<uintptr_t>(T::execute));
    h->mov(abi_param1, reinterpret_cast<uintptr_t>(m_kernel_executor.get()));
    h->mov(abi_param2, h->rsp);

    spill.rsp_align(callee_saved_reg.getIdx());
    h->call(aux_reg);
    spill.rsp_restore();

    h->add(h->rsp, reserved_stack_size);

    spill.postamble();
}

}  // namespace ov::intel_cpu
