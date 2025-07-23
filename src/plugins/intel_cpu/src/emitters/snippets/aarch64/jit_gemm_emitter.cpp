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
    m_buffer_ids = {ov::intel_cpu::emitters::snippets::utils::get_buffer_cluster_id(expr->get_input_port(0)),
                    ov::intel_cpu::emitters::snippets::utils::get_buffer_cluster_id(expr->get_input_port(1)),
                    ov::intel_cpu::emitters::snippets::utils::get_buffer_cluster_id(expr->get_output_port(0))};
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

    // Apply memory offsets to input/output pointers
    const auto sp_size = rnd_up(3 * get_gpr_length(), sp_alignment);
    h->sub(h->sp, h->sp, sp_size);
    // Apply offsets and store pointers on stack
    for (size_t i = 0; i < mem_ptrs.size(); i++) {
        const auto& ptr_reg = mem_ptrs[i];
        int32_t stack_offset = i * get_gpr_length();
        if (ov::snippets::utils::is_dynamic_value(m_memory_offsets[i])) {
            // Dynamic offset: read from runtime parameters
            size_t runtime_offset = GET_OFF(buffer_offsets) + m_buffer_ids[i] * sizeof(size_t);
            utils::push_ptr_with_runtime_offset_on_stack(h, stack_offset, ptr_reg, aux_reg, runtime_offset);
        } else {
            // Static offset: add compile-time constant
            utils::push_ptr_with_static_offset_on_stack(h, stack_offset, ptr_reg, m_memory_offsets[i]);
        }
    }
    // Load back the adjusted pointers for function call
    h->ldr(x1, Xbyak_aarch64::ptr(h->sp));                        // matrix A (in0)
    h->ldr(x2, Xbyak_aarch64::ptr(h->sp, get_gpr_length()));      // matrix B (in1)
    h->ldr(x3, Xbyak_aarch64::ptr(h->sp, 2 * get_gpr_length()));  // matrix C (out)

    // Restore stack pointer
    h->add(h->sp, h->sp, sp_size);

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
