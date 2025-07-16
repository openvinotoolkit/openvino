// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_gemm_emitter.hpp"

#include <cpu/aarch64/xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_adr.h>
#include <cpu/aarch64/xbyak_aarch64/xbyak_aarch64/xbyak_aarch64_reg.h>

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include "emitters/snippets/aarch64/kernel_executors/gemm.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"

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
}

std::set<std::vector<element::Type>> jit_gemm_emitter::get_supported_precisions(
    [[maybe_unused]] const std::shared_ptr<ov::Node>& node) {
    // Note: currently supports only fp32 on arm
    return {{element::f32, element::f32}};
}

void jit_gemm_emitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    OV_CPU_JIT_EMITTER_ASSERT(in.size() == 2, "Expects 2 input regs, got", in.size());
    OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Expects 1 output reg, got", out.size());
}

void jit_gemm_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    validate_arguments(in, out);
    // todo: use optimized reg spill after CVS-162498
    std::unordered_set<size_t> exclude = {};
    store_context(exclude);

    auto get_free_scratch_reg = [&](const std::vector<size_t>& in_use) -> size_t {
        for (size_t reg = 19; reg <= 28; ++reg) {
            bool is_free = true;
            for (size_t used_reg : in_use) {
                if (reg == used_reg) {
                    is_free = false;
                    break;
                }
            }
            if (is_free) {
                return reg;
            }
        }
        OV_CPU_JIT_EMITTER_THROW("No free scratch register available");
    };

    std::vector<size_t> used_regs = {in[0], in[1], out[0]};
    auto temp_a = get_free_scratch_reg(used_regs);
    used_regs.push_back(temp_a);
    auto temp_b = get_free_scratch_reg(used_regs);
    used_regs.push_back(temp_b);
    auto temp_dst = get_free_scratch_reg(used_regs);

    h->mov(Xbyak_aarch64::XReg(temp_a), Xbyak_aarch64::XReg(in[0]));
    h->mov(Xbyak_aarch64::XReg(temp_b), Xbyak_aarch64::XReg(in[1]));
    h->mov(Xbyak_aarch64::XReg(temp_dst), Xbyak_aarch64::XReg(out[0]));

    h->mov(Xbyak_aarch64::XReg(1), Xbyak_aarch64::XReg(temp_a));
    h->mov(Xbyak_aarch64::XReg(2), Xbyak_aarch64::XReg(temp_b));
    h->mov(Xbyak_aarch64::XReg(3), Xbyak_aarch64::XReg(temp_dst));
    h->mov(Xbyak_aarch64::XReg(0), get_compiled_kernel_ptr());
    h->mov(Xbyak_aarch64::XReg(18), get_execute_function_ptr());
    h->blr(Xbyak_aarch64::XReg(18));

    restore_context(exclude);
}

const uintptr_t jit_gemm_emitter::get_compiled_kernel_ptr() const {
    return reinterpret_cast<const uintptr_t>(m_kernel_executor_kai.get());
}

const uintptr_t jit_gemm_emitter::get_execute_function_ptr() {
    return reinterpret_cast<const uintptr_t>(GemmKaiKernelExecutor::execute);
}

}  // namespace ov::intel_cpu::aarch64
