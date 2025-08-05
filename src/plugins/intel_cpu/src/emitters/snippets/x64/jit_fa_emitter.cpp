// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_fa_emitter.hpp"

#include <xbyak/xbyak.h>

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "cache/multi_cache.h"
#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/plugin/x64/utils.hpp"
#include "emitters/snippets/x64/jit_binary_call_emitter.hpp"
#include "emitters/snippets/x64/kernel_executors/fa.hpp"
#include "emitters/snippets/x64/utils.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "transformations/snippets/x64/op/fa.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace ov::intel_cpu::x64;

namespace ov::intel_cpu {

jit_fa_emitter::jit_fa_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                               dnnl::impl::cpu::x64::cpu_isa_t isa,
                               const ov::snippets::lowered::ExpressionPtr& expr,
                               const snippets::KernelExecutorTablePtr& kernel_table,
                               const ov::intel_cpu::MultiCacheWeakPtr& compiled_kernel_cache)
    : jit_binary_call_emitter(h, isa, expr->get_live_regs()) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto fa_node = ov::as_type_ptr<FACPU>(expr->get_node());
    auto fa_config = fa_node->get_config();
    const FAKernelConfig kernel_config(fa_config);
    m_kernel_executor_fa = kernel_table->register_kernel<FAKernelExecutor>(expr, compiled_kernel_cache, kernel_config);
    m_memory_offsets = {fa_node->get_offset_a(),
                        fa_node->get_offset_b(),
                        fa_node->get_offset_c(),
                        fa_node->get_offset_d()};
}

std::set<std::vector<element::Type>> jit_fa_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    // Note: currently supports only fp32
    return {{element::f32, element::f32, element::f32}};
}

void jit_fa_emitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == 3, "Expects 3 input regs, got", in.size());
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Expects 1 output reg, got", out.size());
}

uintptr_t jit_fa_emitter::get_compiled_kernel_ptr() const {
    return reinterpret_cast<const uintptr_t>(m_kernel_executor_fa.get());
}

uintptr_t jit_fa_emitter::get_execute_function_ptr() {
    return reinterpret_cast<const uintptr_t>(ov::intel_cpu::x64::FAKernelExecutor::execute);
}

void jit_fa_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    std::vector<size_t> mem_ptrs_idxs{in[0], in[1], in[2], out[0]};
    init_binary_call_regs(2, mem_ptrs_idxs);

    const Xbyak::Reg64& aux_reg = get_call_address_reg();
    const Xbyak::Reg64& callee_saved_reg = get_callee_saved_reg();

    EmitABIRegSpills spill(h);
    // save abi_params, and callee_saved_reg. aux_reg saved in base jit_emitter. all could be changed in call
    spill.preamble(get_regs_to_spill());

    auto reserved_stack_size = sizeof(FAKernelExecutor::call_args);
    // Reserve memory on the stack
    h->sub(h->rsp, reserved_stack_size);

#define GET_OFF_CALL_ARGS(field) offsetof(FAKernelExecutor::call_args, field)
    const std::vector<size_t> fa_args_offsets = {GET_OFF_CALL_ARGS(A),
                                                 GET_OFF_CALL_ARGS(B),
                                                 GET_OFF_CALL_ARGS(C),
                                                 GET_OFF_CALL_ARGS(D)};

    const auto& mem_ptrs = ov::intel_cpu::utils::transform_idxs_to_regs(mem_ptrs_idxs);
    for (size_t i = 0; i < mem_ptrs_idxs.size(); i++) {
        utils::push_ptr_with_static_offset_on_stack(h, fa_args_offsets[i], mem_ptrs[i], m_memory_offsets[i]);
    }
#undef GET_OFF_CALL_ARGS

    h->mov(aux_reg, get_execute_function_ptr());
    h->mov(abi_param1, get_compiled_kernel_ptr());
    // move from satck to abi_param2
    h->mov(abi_param2, h->rsp);

    spill.rsp_align(callee_saved_reg.getIdx());
    h->call(aux_reg);
    spill.rsp_restore();

    h->add(h->rsp, reserved_stack_size);

    spill.postamble();
}

}  // namespace ov::intel_cpu
