// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_gemm_emitter.hpp"

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_adr.h>
#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_reg.h>

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include "emitters/snippets/aarch64/kernel_executors/gemm.hpp"
#include "emitters/snippets/aarch64/utils.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/utils/utils.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/aarch64/op/gemm_cpu.hpp"

using namespace Xbyak_aarch64;

namespace ov::intel_cpu::aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

jit_gemm_emitter::jit_gemm_emitter(jit_generator* h,
                                   cpu_isa_t isa,
                                   const ExpressionPtr& expr,
                                   const snippets::KernelExecutorTablePtr& kernel_table)
    : jit_emitter(h, isa) {
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
    // todo: use optimized reg spill after CVS-162498
    std::unordered_set<size_t> exclude = {};
    store_context(exclude);

    Xbyak_aarch64::XReg x0(0);
    Xbyak_aarch64::XReg x1(1);
    Xbyak_aarch64::XReg x2(2);
    Xbyak_aarch64::XReg x3(3);
    Xbyak_aarch64::XReg aux_reg(5);

    // Prepare memory pointers with offsets
    std::vector<size_t> mem_ptrs_idxs{in[0], in[1], out[0]};
    const auto& mem_ptrs = utils::transform_idxs_to_regs(mem_ptrs_idxs);

    // Apply memory offsets and load adjusted pointers
    std::vector<Xbyak_aarch64::XReg> load_regs{x1, x2, x3};

    // Dynamically choose safe auxiliary registers that don't conflict with mem_ptrs or load_regs
    std::vector<size_t> used_indices;
    for (const auto& reg : mem_ptrs)
        used_indices.push_back(reg.getIdx());
    for (const auto& reg : load_regs)
        used_indices.push_back(reg.getIdx());
    std::vector<Xbyak_aarch64::XReg> aux_regs = utils::get_aux_gprs(used_indices);

    utils::push_and_load_ptrs_with_offsets(h, mem_ptrs, m_memory_offsets, m_buffer_ids, aux_regs, load_regs);

    // Set up executor pointer as first argument
    const auto& compiled_kernel = get_compiled_kernel_ptr();
    h->mov(x0, compiled_kernel);

    Xbyak_aarch64::XReg func_reg(9);
    h->mov(func_reg, get_execute_function_ptr());
    h->blr(func_reg);

    restore_context(exclude);
}

const uintptr_t jit_gemm_emitter::get_compiled_kernel_ptr() const {
    return reinterpret_cast<const uintptr_t>(m_kernel_executor_kai.get());
}

const uintptr_t jit_gemm_emitter::get_execute_function_ptr() {
    return reinterpret_cast<const uintptr_t>(GemmKaiKernelExecutor::execute);
}

}  // namespace ov::intel_cpu::aarch64
