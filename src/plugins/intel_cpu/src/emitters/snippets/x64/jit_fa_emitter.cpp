// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_fa_emitter.hpp"

#include <xbyak/xbyak.h>

#include <common/utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/plugin/x64/utils.hpp"
#include "emitters/snippets/x64/utils.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/fa.hpp"
#include "utils/general_utils.h"
#include "emitters/snippets/x64/kernel_executors/fa.hpp"
#include "transformations/snippets/x64/op/fa.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace ov::intel_cpu::x64;

namespace ov::intel_cpu {

jit_fa_emitter::jit_fa_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                               dnnl::impl::cpu::x64::cpu_isa_t isa,
                               const ov::snippets::lowered::ExpressionPtr& expr,
                               const snippets::KernelExecutorTablePtr& kernel_table)
    : jit_binary_call_emitter(h, isa, expr->get_live_regs()) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto fa_node = ov::as_type_ptr<FACPU>(expr->get_node());
    auto fa_config = fa_node->get_config();
    const FAKernelConfig kernel_config(fa_config);
    m_kernel_executor_fa = kernel_table->register_kernel<FAKernelExecutor>(expr, kernel_config);
    m_memory_offsets = {fa_node->get_offset_a(), fa_node->get_offset_b(), fa_node->get_offset_c(), fa_node->get_offset_d()};
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

const uintptr_t jit_fa_emitter::get_compiled_kernel_ptr() const {
    return reinterpret_cast<const uintptr_t>(m_kernel_executor_fa.get());
}

const uintptr_t jit_fa_emitter::get_execute_function_ptr() {
    return reinterpret_cast<const uintptr_t>(ov::intel_cpu::x64::FAKernelExecutor::execute);
}

void jit_fa_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    std::vector<size_t> mem_ptrs_idxs{in[0], in[1], in[2], out[0]};
    init_binary_call_regs(5, mem_ptrs_idxs);  // should 3(in)+1(out)+1(executor kernel ptr as abi_param1)

    const Xbyak::Reg64& aux_reg = get_call_address_reg();
    const Xbyak::Reg64& callee_saved_reg = get_callee_saved_reg();

    EmitABIRegSpills spill(h);
    // save 5 abi_params, and callee_saved_reg. aux_reg saved in base jit_emitter. all are changed in call
    spill.preamble(get_regs_to_spill());

    // move in/out registers to abi_params, to avoid overwrite in/out registers, need move to stack, then to abi_params
    size_t reserved_stack_size = mem_ptrs_idxs.size() * 8;
    h->sub(h->rsp, reserved_stack_size);

    const auto& mem_ptrs = ov::intel_cpu::utils::transform_idxs_to_regs(mem_ptrs_idxs);
    for (size_t i = 0; i < mem_ptrs_idxs.size(); i++) {
        utils::push_ptr_with_static_offset_on_stack(h, i * 8, mem_ptrs[i], m_memory_offsets[i]);
    }

    h->mov(aux_reg, reinterpret_cast<uintptr_t>(get_execute_function_ptr()));
    h->mov(abi_param1, reinterpret_cast<uintptr_t>(get_compiled_kernel_ptr()));
    // move from satck to abi_param2-5
    h->mov(abi_param2, h->ptr[h->rsp]);
    h->mov(abi_param3, h->ptr[h->rsp + 8]);
    h->mov(abi_param4, h->ptr[h->rsp + 2 * 8]);
    h->mov(abi_param5, h->ptr[h->rsp + 3 * 8]);

    h->add(h->rsp, reserved_stack_size);

    spill.rsp_align(callee_saved_reg.getIdx());
    h->call(aux_reg);
    spill.rsp_restore();

    spill.postamble();
}

}  // namespace ov::intel_cpu
