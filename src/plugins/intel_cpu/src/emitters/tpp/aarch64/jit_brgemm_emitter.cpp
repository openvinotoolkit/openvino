// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_brgemm_emitter.hpp"

#include "snippets/utils/utils.hpp"
#include "transformations/tpp/common/op/brgemm.hpp"

using namespace ov::intel_cpu::tpp;
using namespace Xbyak_aarch64;

namespace ov::intel_cpu::aarch64 {

using jit_generator = dnnl::impl::cpu::aarch64::jit_generator;
using cpu_isa_t = dnnl::impl::cpu::aarch64::cpu_isa_t;
using ExpressionPtr = ov::snippets::lowered::ExpressionPtr;

jit_brgemm_emitter::jit_brgemm_emitter(jit_generator* h,
                                       cpu_isa_t isa,
                                       const ExpressionPtr& expr,
                                       const snippets::KernelExecutorTablePtr& kernel_table,
                                       const ov::intel_cpu::MultiCacheWeakPtr& compiled_kernel_cache)
    : jit_emitter(h, isa) {
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
    const auto& brgemm_node = as_type_ptr<intel_cpu::tpp::op::BrgemmTPP>(expr->get_node());
    const auto& brg0Prc = brgemm_node->get_input_element_type(0);
    const auto& brg1Prc = brgemm_node->get_input_element_type(1);
    BrgemmKernelConfig kernel_config(brg0Prc, brg1Prc);
    m_kernel_executor = kernel_table->register_kernel<BrgemmKernelExecutor>(expr, compiled_kernel_cache, kernel_config);
}

std::set<std::vector<element::Type>> jit_brgemm_emitter::get_supported_precisions(
    const std::shared_ptr<ov::Node>& node) {
    // Note: Brgemm currently supports only fp32 on arm
    return {{element::f32, element::f32}};
}

void jit_brgemm_emitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == 2, "Expects 2 input regs, got", in.size());
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Expects 1 output reg, got", out.size());
}

void jit_brgemm_emitter::emit_code_impl(const std::vector<size_t>& in,
                                        const std::vector<size_t>& out,
                                        const std::vector<size_t>& pool_vec_idxs,
                                        const std::vector<size_t>& pool_gpr_idxs) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void jit_brgemm_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    // todo: use optimized reg spill after CVS-162498
    std::unordered_set<size_t> exclude = {};
    store_context(exclude);

    Xbyak_aarch64::XReg func_reg(9);
    h->mov(func_reg, get_execute_function_ptr());
    Xbyak_aarch64::XReg x0(0);
    Xbyak_aarch64::XReg x1(1);
    Xbyak_aarch64::XReg x2(2);
    Xbyak_aarch64::XReg x3(3);

    const auto& compiled_kernel = get_compiled_kernel_ptr();
    h->mov(x0, compiled_kernel);
    h->mov(x1, Xbyak_aarch64::XReg(in[0]));
    h->mov(x2, Xbyak_aarch64::XReg(in[1]));
    h->mov(x3, Xbyak_aarch64::XReg(out[0]));
    h->blr(func_reg);

    restore_context(exclude);
}

const uintptr_t jit_brgemm_emitter::get_compiled_kernel_ptr() const {
    return reinterpret_cast<const uintptr_t>(m_kernel_executor.get());
}

const uintptr_t jit_brgemm_emitter::get_execute_function_ptr() const {
    return reinterpret_cast<const uintptr_t>(BrgemmKernelExecutor::execute);
}

}  // namespace ov::intel_cpu::aarch64
