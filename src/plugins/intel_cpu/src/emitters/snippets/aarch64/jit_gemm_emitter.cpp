// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_gemm_emitter.hpp"

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include "emitters/snippets/aarch64/jit_binary_call_emitter.hpp"
#include "emitters/snippets/aarch64/kernel_executors/gemm.hpp"
#include "emitters/snippets/aarch64/utils.hpp"
#include "emitters/snippets/utils/utils.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"
#include "utils/general_utils.h"

using namespace Xbyak_aarch64;

namespace ov::intel_cpu::aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

jit_gemm_emitter::jit_gemm_emitter(jit_generator* h,
                                   cpu_isa_t isa,
                                   const ExpressionPtr& expr,
                                   const snippets::KernelExecutorTablePtr& kernel_table)
    : jit_binary_call_emitter(h, isa, expr->get_live_regs()) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    GemmKernelKaiConfig kernel_config;
    m_kernel_executor_kai = kernel_table->register_kernel<GemmKaiKernelExecutor>(expr, kernel_config);

    const auto gemm_node = as_type_ptr<GemmCPU>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(gemm_node, "Expected GemmCPU node");

    // Initialize memory offsets similar to x64 brgemm implementation
    m_memory_offsets = {gemm_node->get_offset_a(), gemm_node->get_offset_b(), gemm_node->get_offset_c()};

    // Initialize buffer IDs using the utils function
    m_buffer_ids = {ov::intel_cpu::utils::get_buffer_cluster_id(expr->get_input_port(0)),
                    ov::intel_cpu::utils::get_buffer_cluster_id(expr->get_input_port(1)),
                    ov::intel_cpu::utils::get_buffer_cluster_id(expr->get_output_port(0))};
}

std::set<std::vector<element::Type>> jit_gemm_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    // Note: currently supports only fp32 on arm
    return {{element::f32, element::f32}};
}

void jit_gemm_emitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == 2, "Expects 2 input regs, got", in.size());
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Expects 1 output reg, got", out.size());
    OV_CPU_JIT_EMITTER_ASSERT(m_memory_offsets.size() == 3, "Expected 3 memory offsets for A, B, C");
    OV_CPU_JIT_EMITTER_ASSERT(m_buffer_ids.size() == 3, "Expected 3 buffer IDs for A, B, C");
}

void jit_gemm_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);

    std::vector<size_t> mem_ptrs_idxs{in[0], in[1], out[0]};

    init_binary_call_regs(2, mem_ptrs_idxs);
    emit_call(mem_ptrs_idxs);
}

void jit_gemm_emitter::emit_call(const std::vector<size_t>& mem_ptrs_idxs) const {
    const auto& call_address_reg = get_call_address_reg();
    std::unordered_set<size_t> exclude_spill = {};
    store_context(exclude_spill);

    auto reserved_stack_size = ov::intel_cpu::rnd_up(sizeof(GemmKaiKernelExecutor::call_args), sp_alignment);
    emit_stack_preserve(reserved_stack_size);

    const size_t A_offset = offsetof(GemmKaiKernelExecutor::call_args, A);
    const size_t B_offset = offsetof(GemmKaiKernelExecutor::call_args, B);
    const size_t C_offset = offsetof(GemmKaiKernelExecutor::call_args, C);

    const std::vector<size_t> gemm_args_offsets = {A_offset, B_offset, C_offset};

    const auto& mem_ptrs = utils::transform_idxs_to_regs(mem_ptrs_idxs);

    // Collect used register indices to avoid conflicts with auxiliary registers
    std::vector<size_t> used_gpr_idxs = {call_address_reg.getIdx()};
    for (const auto& reg : mem_ptrs) {
        used_gpr_idxs.push_back(reg.getIdx());
    }

    // Get auxiliary registers for the helper function (needs at least 3 for dynamic offsets)
    auto aux_gprs = ov::intel_cpu::aarch64::utils::get_aux_gprs(used_gpr_idxs, 3);

    // Use the new helper function to push all pointers with offsets to their stack locations
    utils::push_ptrs_with_offsets_to_stack(h, mem_ptrs, m_memory_offsets, m_buffer_ids, aux_gprs, gemm_args_offsets);

    // Note: scratch field was removed per earlier review feedback, so we don't need to zero it

    Xbyak_aarch64::XReg x0(0);
    Xbyak_aarch64::XReg x1(1);

    h->mov(call_address_reg, reinterpret_cast<uintptr_t>(GemmKaiKernelExecutor::execute));

    h->mov(x0, reinterpret_cast<uintptr_t>(m_kernel_executor_kai.get()));
    h->mov(x1, h->sp);

    h->blr(call_address_reg);

    emit_stack_restore(reserved_stack_size);

    restore_context(exclude_spill);
}

uintptr_t jit_gemm_emitter::get_compiled_kernel_ptr() const {
    return reinterpret_cast<const uintptr_t>(m_kernel_executor_kai.get());
}

uintptr_t jit_gemm_emitter::get_execute_function_ptr() {
    return reinterpret_cast<const uintptr_t>(GemmKaiKernelExecutor::execute);
}

}  // namespace ov::intel_cpu::aarch64
