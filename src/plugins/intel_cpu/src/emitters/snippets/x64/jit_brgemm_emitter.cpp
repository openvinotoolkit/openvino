// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_brgemm_emitter.hpp"

#include <xbyak/xbyak.h>

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <type_traits>
#include <vector>

#include "cache/multi_cache.h"
#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/plugin/x64/utils.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/utils/utils.hpp"
#include "emitters/snippets/x64/jit_binary_call_emitter.hpp"
#include "emitters/snippets/x64/kernel_executors/brgemm.hpp"
#include "emitters/snippets/x64/kernel_executors/brgemm_amx.hpp"
#include "emitters/snippets/x64/kernel_executors/brgemm_base.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "transformations/snippets/x64/op/brgemm_utils.hpp"
#include "utils.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace ov::intel_cpu::x64;

namespace ov::intel_cpu {

jit_brgemm_emitter::jit_brgemm_emitter(jit_generator_t* h,
                                       cpu_isa_t isa,
                                       const ov::snippets::lowered::ExpressionPtr& expr,
                                       const snippets::KernelExecutorTablePtr& kernel_table,
                                       const ov::intel_cpu::MultiCacheWeakPtr& compiled_kernel_cache)
    : jit_emitter(h, isa),
      jit_binary_call_emitter(h, isa, expr->get_live_regs()) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto& brgemm_node = as_type_ptr<ov::intel_cpu::BrgemmCPU>(expr->get_node());
    const auto& brgOutPrc = brgemm_node->get_output_element_type(0);
    const auto& brgemm_config = brgemm_node->get_config();
    const auto& post_ops_config = brgemm_node->get_postops_config();
    m_binary_postops_offset = post_ops_config.binary_postops_offset;

    if (brgemm_config.is_amx()) {
        BrgemmAMXKernelConfig config(brgemm_config, brgOutPrc, post_ops_config.post_ops);
        m_kernel_executor = kernel_table->register_kernel<BrgemmAMXKernelExecutor>(expr, compiled_kernel_cache, config);
    } else {
        BrgemmKernelConfig config(brgemm_config, brgOutPrc, post_ops_config.post_ops);
        m_kernel_executor = kernel_table->register_kernel<BrgemmKernelExecutor>(expr, compiled_kernel_cache, config);
    }
    // Note: even if the Brgemm node is dynamic, the first shapeInfer and RuntimeConfigurator::update()
    // are performed before the BrgemmKernelExecutor registration. So we have to trigger update() manually
    // for both static and the 1st dynamic shapes.
    OV_CPU_JIT_EMITTER_ASSERT(!snippets::utils::is_dynamic_vdims(expr->get_input_port_descriptor(0)->get_shape()) &&
                                  !snippets::utils::is_dynamic_vdims(expr->get_input_port_descriptor(1)->get_shape()),
                              "Jit emitter is called when the shapes are unknown");

    m_memory_offsets = {brgemm_node->get_offset_a(), brgemm_node->get_offset_b(), brgemm_node->get_offset_c()};
    m_buffer_ids = {ov::intel_cpu::utils::get_buffer_cluster_id(expr->get_input_port(0)),
                    ov::intel_cpu::utils::get_buffer_cluster_id(expr->get_input_port(1)),
                    ov::intel_cpu::utils::get_buffer_cluster_id(expr->get_output_port(0))};
    m_with_scratchpad = brgemm_config.with_scratchpad();
    if (m_with_scratchpad) {
        m_memory_offsets.push_back(brgemm_node->get_offset_scratch());
        m_buffer_ids.push_back(ov::intel_cpu::utils::get_buffer_cluster_id(expr->get_input_port(2)));
    }
    m_gemm_inputs_count = brgemm_node->get_gemm_inputs_count();
}

std::set<std::vector<element::Type>> jit_brgemm_emitter::get_supported_precisions(
    const std::shared_ptr<ov::Node>& node) {
    const auto brgemm = as_type_ptr<ov::intel_cpu::BrgemmCPU>(node);
    OV_CPU_JIT_EMITTER_ASSERT(brgemm, "get_supported_precisions() expects BrgemmCPU node");
    const auto& config = brgemm->get_config();

    auto form_precisions = [&brgemm](const element::TypeVector& precisions) {
        OPENVINO_ASSERT(precisions.size() == brgemm->get_gemm_inputs_count(),
                        "precisions size should be equal to the number of gemm inputs");
        auto res = precisions;
        // Note: all postops are supported only in f32 precision
        for (size_t i = brgemm->get_gemm_inputs_count(); i < brgemm->input_values().size(); ++i) {
            res.push_back(element::f32);
        }
        return res;
    };
    if (config.is_amx()) {
        std::set<std::vector<element::Type>> supported_types = {
            form_precisions({element::i8, element::i8, element::u8}),
            form_precisions({element::u8, element::i8, element::u8}),
            form_precisions({element::bf16, element::bf16, element::u8})};
        if (config.isa() == dnnl::impl::cpu::x64::avx512_core_amx_fp16) {
            supported_types.insert(form_precisions({element::f16, element::f16, element::u8}));
        }
        return supported_types;
    }
    if (config.with_compensations()) {
        return {form_precisions({element::i8, element::i8, element::f32})};
    }
    if (config.with_wei_repacking()) {
        std::set<std::vector<element::Type>> supported_types = {form_precisions({element::f32, element::f32})};
        if (snippets::utils::any_of(config.isa(),
                                    dnnl::impl::cpu::x64::avx512_core_bf16,
                                    dnnl::impl::cpu::x64::avx2_vnni_2)) {
            supported_types.insert(form_precisions({element::bf16, element::bf16}));
        }
        if (snippets::utils::any_of(config.isa(),
                                    dnnl::impl::cpu::x64::avx512_core_vnni,
                                    dnnl::impl::cpu::x64::avx2_vnni)) {
            supported_types.insert(form_precisions({element::u8, element::i8}));
        }
        if (config.isa() == dnnl::impl::cpu::x64::avx2_vnni_2) {
            supported_types.insert(form_precisions({element::i8, element::i8}));
            supported_types.insert(form_precisions({element::f16, element::f16}));
        }
        return supported_types;
    }
    return {form_precisions({element::f32, element::f32})};
}

void jit_brgemm_emitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(m_memory_offsets.size() == m_gemm_inputs_count + 1, "invalid memory offsets size");
    OV_CPU_JIT_EMITTER_ASSERT(in.size() >= m_gemm_inputs_count && out.size() == 1,
                              "expects 3 inputs if there are compensations/wsp");
}

void jit_brgemm_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    std::vector<size_t> mem_ptrs_idxs{in[0], in[1], out[0]};
    init_binary_call_regs(2, mem_ptrs_idxs);
    if (m_with_scratchpad) {
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

    // Prepare external pointers
    if (m_binary_postops_offset) {
        h->mov(aux_reg, h->ptr[abi_param1 + GET_OFF(external_ptrs)]);
        h->add(aux_reg, m_binary_postops_offset.value() * sizeof(void**));
        h->mov(h->qword[h->rsp + GET_OFF_CALL_ARGS(post_ops_binary_arg_vec)], aux_reg);
    } else {
        h->mov(h->qword[h->rsp + GET_OFF_CALL_ARGS(post_ops_binary_arg_vec)], reinterpret_cast<uintptr_t>(nullptr));
    }
#undef GET_OFF_CALL_ARGS

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
