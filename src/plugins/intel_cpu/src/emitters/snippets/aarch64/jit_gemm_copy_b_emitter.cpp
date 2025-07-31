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
}

std::set<std::vector<element::Type>> jit_gemm_copy_b_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    // Note: Brgemm currently supports only fp32 on arm
    return {{element::f32}};
}

void jit_gemm_copy_b_emitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == 1, "Expects 1 input reg, got", in.size());
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Expects 1 output reg, got", out.size());
}

void jit_gemm_copy_b_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    // todo: use optimized reg spill after CVS-162498
    std::unordered_set<size_t> exclude = {};
    store_context(exclude);

    Xbyak_aarch64::XReg x0(0);
    Xbyak_aarch64::XReg x1(1);
    Xbyak_aarch64::XReg x2(2);
    h->str(Xbyak_aarch64::XReg(in[0]), pre_ptr(h->sp, -get_vec_length()));
    h->str(Xbyak_aarch64::XReg(out[0]), pre_ptr(h->sp, -get_vec_length()));
    h->ldr(x2, post_ptr(h->sp, get_vec_length()));
    h->ldr(x1, post_ptr(h->sp, get_vec_length()));
    const auto& compiled_kernel = get_compiled_kernel_ptr();
    h->mov(x0, compiled_kernel);

    Xbyak_aarch64::XReg func_reg(9);
    h->mov(func_reg, get_execute_function_ptr());
    h->blr(func_reg);

    restore_context(exclude);
}

uintptr_t jit_gemm_copy_b_emitter::get_compiled_kernel_ptr() const {
    return reinterpret_cast<const uintptr_t>(m_kernel_executor.get());
}

uintptr_t jit_gemm_copy_b_emitter::get_execute_function_ptr() {
    return reinterpret_cast<const uintptr_t>(GemmCopyBKaiKernelExecutor::execute);
}
}  // namespace ov::intel_cpu::aarch64
