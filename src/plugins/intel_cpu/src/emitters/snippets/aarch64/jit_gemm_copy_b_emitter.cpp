// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_gemm_copy_b_emitter.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_reg.h>

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include "emitters/snippets/aarch64/jit_binary_call_emitter.hpp"
#include "emitters/snippets/aarch64/kernel_executors/gemm_copy_b.hpp"
#include "emitters/snippets/aarch64/utils.hpp"
#include "emitters/snippets/utils/utils.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "transformations/snippets/aarch64/op/gemm_copy_b.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

jit_gemm_copy_b_emitter::jit_gemm_copy_b_emitter(jit_generator* h,
                                                 cpu_isa_t isa,
                                                 const ExpressionPtr& expr,
                                                 const snippets::KernelExecutorTablePtr& kernel_table)
    : jit_binary_call_emitter(h, isa, expr->get_live_regs()) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto gemm_repack = ov::as_type_ptr<GemmCopyB>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(gemm_repack, "expects GemmCopyB node");
    m_kernel_executor = kernel_table->register_kernel<GemmCopyBKaiKernelExecutor>(expr, GemmCopyBKernelKaiConfig());

    // Initialize memory offsets similar to x64 brgemm_copy_b implementation
    m_memory_offsets = {gemm_repack->get_offset_in(), gemm_repack->get_offset_bias(), gemm_repack->get_offset_out()};

    // Initialize buffer IDs using the utils function
    m_buffer_ids = {ov::intel_cpu::utils::get_buffer_cluster_id(expr->get_input_port(0)),
                    ov::intel_cpu::utils::get_buffer_cluster_id(expr->get_input_port(1)),
                    ov::intel_cpu::utils::get_buffer_cluster_id(expr->get_output_port(0))};
}

std::set<std::vector<element::Type>> jit_gemm_copy_b_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    // Note: Brgemm currently supports only fp32 on arm
    return {{element::f32, element::u8}};
}

void jit_gemm_copy_b_emitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == 2, "Expects 2 input reg, got", in.size());
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Expects 1 output reg, got", out.size());
    OV_CPU_JIT_EMITTER_ASSERT(m_memory_offsets.size() == 3, "Expected 3 memory offsets for input and output");
    OV_CPU_JIT_EMITTER_ASSERT(m_buffer_ids.size() == 3, "Expected 3 buffer IDs for input and output");
}

void jit_gemm_copy_b_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);

    std::vector<size_t> mem_ptrs_idxs{in[0], in[1], out[0]};

    init_binary_call_regs(2, mem_ptrs_idxs);
    emit_call(mem_ptrs_idxs);
}

void jit_gemm_copy_b_emitter::emit_call(const std::vector<size_t>& mem_ptrs_idxs) const {
    std::unordered_set<size_t> exclude_spill = {};
    store_context(exclude_spill);

    const std::vector<int32_t> args_offsets = {
        static_cast<int32_t>(offsetof(GemmCopyBKaiKernelExecutor::call_args, in)),
        static_cast<int32_t>(offsetof(GemmCopyBKaiKernelExecutor::call_args, bias)),
        static_cast<int32_t>(offsetof(GemmCopyBKaiKernelExecutor::call_args, out))};

    const auto& mem_ptrs = utils::transform_idxs_to_regs(mem_ptrs_idxs);

    std::vector<size_t> used_gpr_idxs;
    used_gpr_idxs.reserve(mem_ptrs.size());
    for (const auto& reg : mem_ptrs) {
        used_gpr_idxs.push_back(reg.getIdx());
    }
    std::vector<Xbyak_aarch64::XReg> aux_regs = utils::get_aux_gprs(used_gpr_idxs);

    auto reserved_stack_size = ov::intel_cpu::rnd_up(sizeof(GemmCopyBKaiKernelExecutor::call_args), sp_alignment);
    emit_stack_preserve(reserved_stack_size);

    utils::push_ptrs_with_offsets_to_stack(h, mem_ptrs, m_memory_offsets, m_buffer_ids, aux_regs, args_offsets);

    const auto& call_address_reg = get_call_address_reg();
    h->mov(call_address_reg, get_execute_function_ptr());

    Xbyak_aarch64::XReg x0(0), x1(1);
    h->mov(x0, get_compiled_kernel_ptr());
    h->mov(x1, h->sp);

    h->blr(call_address_reg);

    emit_stack_restore(reserved_stack_size);

    restore_context(exclude_spill);
}

uintptr_t jit_gemm_copy_b_emitter::get_compiled_kernel_ptr() const {
    return reinterpret_cast<const uintptr_t>(m_kernel_executor.get());
}

uintptr_t jit_gemm_copy_b_emitter::get_execute_function_ptr() {
    return reinterpret_cast<const uintptr_t>(GemmCopyBKaiKernelExecutor::execute);
}
}  // namespace ov::intel_cpu::aarch64