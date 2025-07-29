// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_gemm_copy_b_emitter.hpp"

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

#include "emitters/snippets/aarch64/kernel_executors/gemm_copy_b.hpp"
#include "emitters/snippets/aarch64/utils.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/utils/utils.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/aarch64/op/gemm_copy_b.hpp"
#include "transformations/snippets/aarch64/op/gemm_utils.hpp"

namespace ov::intel_cpu::aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

jit_gemm_copy_b_emitter::jit_gemm_copy_b_emitter(jit_generator* h,
                                                 cpu_isa_t isa,
                                                 const ExpressionPtr& expr,
                                                 const snippets::KernelExecutorTablePtr& kernel_table)
    : jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto gemm_repack = ov::as_type_ptr<GemmCopyB>(expr->get_node());
    OV_CPU_JIT_EMITTER_ASSERT(gemm_repack, "expects GemmCopyB node");
    const auto& child_gemms = ov::intel_cpu::aarch64::gemm_utils::repacking::get_gemm_exprs(expr);
    size_t n_blk_size = 0;
    for (const auto& child_gemm : child_gemms) {
        const auto& gemm_in1_subtensor = ov::snippets::utils::get_projected_subtensor(child_gemm->get_input_port(1));
        const auto& current_block = *gemm_in1_subtensor.rbegin();
        if (current_block != snippets::utils::get_dynamic_value<size_t>() && current_block > n_blk_size) {
            n_blk_size = current_block;
        }
    }
    OV_CPU_JIT_EMITTER_ASSERT(n_blk_size > 0, "n_blk_size of gemm_repack is expected to be greater than 0.");
    GemmCopyBKernelKaiConfig kernel_config(n_blk_size);
    m_kernel_executor = kernel_table->register_kernel<GemmCopyBKaiKernelExecutor>(expr, kernel_config);

    // Initialize memory offsets similar to x64 brgemm_copy_b implementation
    m_memory_offsets = {gemm_repack->get_offset_in(), gemm_repack->get_offset_out()};

    // Initialize buffer IDs using the utils function
    m_buffer_ids = {ov::intel_cpu::utils::get_buffer_cluster_id(expr->get_input_port(0)),
                    ov::intel_cpu::utils::get_buffer_cluster_id(expr->get_output_port(0))};
}

std::set<std::vector<element::Type>> jit_gemm_copy_b_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    // Note: Brgemm currently supports only fp32 on arm
    return {{element::f32}};
}

void jit_gemm_copy_b_emitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == 1, "Expects 1 input reg, got", in.size());
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Expects 1 output reg, got", out.size());
    OV_CPU_JIT_EMITTER_ASSERT(m_memory_offsets.size() == 2, "Expected 2 memory offsets for input and output");
    OV_CPU_JIT_EMITTER_ASSERT(m_buffer_ids.size() == 2, "Expected 2 buffer IDs for input and output");
}

void jit_gemm_copy_b_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    // todo: use optimized reg spill after CVS-162498
    std::unordered_set<size_t> exclude = {};
    store_context(exclude);

    Xbyak_aarch64::XReg x0(0);
    Xbyak_aarch64::XReg x1(1);
    Xbyak_aarch64::XReg x2(2);
    Xbyak_aarch64::XReg aux_reg(3);

    // Prepare memory pointers with offsets
    std::vector<size_t> mem_ptrs_idxs{in[0], out[0]};
    const auto& mem_ptrs = utils::transform_idxs_to_regs(mem_ptrs_idxs);

    // Apply memory offsets and load adjusted pointers
    std::vector<Xbyak_aarch64::XReg> load_regs{x1, x2};

    // Dynamically choose safe auxiliary registers that don't conflict with mem_ptrs or load_regs
    std::vector<size_t> used_indices;
    for (const auto& reg : mem_ptrs)
        used_indices.push_back(reg.getIdx());
    for (const auto& reg : load_regs)
        used_indices.push_back(reg.getIdx());
    used_indices.push_back(0);  // X0
    used_indices.push_back(9);  // X9 - func_reg

    std::vector<Xbyak_aarch64::XReg> aux_regs;
    for (size_t i = 2; i <= 30 && aux_regs.size() < 3; ++i) {
        if (std::find(used_indices.begin(), used_indices.end(), i) == used_indices.end()) {
            aux_regs.emplace_back(static_cast<int>(i));
        }
    }

    utils::push_and_load_ptrs_with_offsets(h, mem_ptrs, m_memory_offsets, m_buffer_ids, aux_regs, load_regs);

    // Set up executor pointer as first argument
    const auto& compiled_kernel = get_compiled_kernel_ptr();
    h->mov(x0, compiled_kernel);

    Xbyak_aarch64::XReg func_reg(9);
    h->mov(func_reg, get_execute_function_ptr());
    h->blr(func_reg);

    restore_context(exclude);
}

const uintptr_t jit_gemm_copy_b_emitter::get_compiled_kernel_ptr() const {
    return reinterpret_cast<const uintptr_t>(m_kernel_executor.get());
}

const uintptr_t jit_gemm_copy_b_emitter::get_execute_function_ptr() {
    return reinterpret_cast<const uintptr_t>(GemmCopyBKaiKernelExecutor::execute);
}
}  // namespace ov::intel_cpu::aarch64
