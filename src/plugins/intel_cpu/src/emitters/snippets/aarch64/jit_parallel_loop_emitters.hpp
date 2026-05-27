// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xbyak_aarch64/xbyak_aarch64/xbyak_aarch64.h>

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "emitters/snippets/aarch64/jit_binary_call_emitter.hpp"
#include "emitters/snippets/aarch64/jit_loop_base_emitters.hpp"
#include "emitters/snippets/aarch64/kernel_executors/parallel_loop.hpp"
#include "emitters/snippets/aarch64/utils.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "snippets/kernel_executor_table.hpp"
#include "snippets/lowered/expression.hpp"

namespace ov::intel_cpu::aarch64 {

class jit_parallel_loop_begin_emitter : public jit_loop_begin_base_emitter, public jit_binary_call_emitter {
public:
    jit_parallel_loop_begin_emitter(dnnl::impl::cpu::aarch64::jit_generator_t* h,
                                    dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                                    const ov::snippets::lowered::ExpressionPtr& expr,
                                    const snippets::KernelExecutorTablePtr& kernel_table);

    std::shared_ptr<EmitABIRegSpills> get_parallel_section_reg_spiller() const {
        return m_parallel_section_reg_spiller;
    }

    size_t get_aux_gprs_count() const override {
        return jit_binary_call_emitter::get_aux_gprs_count() + jit_loop_begin_base_emitter::get_aux_gprs_count();
    }

    size_t get_inputs_count() const override {
        return jit_loop_begin_base_emitter::get_inputs_count();
    }

protected:
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override {
        jit_loop_begin_base_emitter::validate_arguments(in, out);
    }

    void emit_code_impl(const std::vector<size_t>& in_idxs,
                        const std::vector<size_t>& out_idxs,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override {
        jit_loop_begin_base_emitter::emit_code_impl(in_idxs, out_idxs, pool_vec_idxs, pool_gpr_idxs);
    }

    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    void emit_parallel_executor_call(std::vector<Xbyak_aarch64::Reg>& used_regs) const;
    void emit_parallel_region_initialization(const std::vector<Xbyak_aarch64::Reg>& regs_to_restore,
                                             size_t work_amount_reg_idx) const;
    std::set<snippets::Reg> get_regs_to_spill_except_mem_ptr_regs() const;

    bool m_is_dynamic = false;
    std::vector<size_t> m_mem_ptr_regs_idxs;
    jit_snippets_call_args::loop_args_t m_loop_args;
    std::shared_ptr<Xbyak_aarch64::Label> m_loop_preamble_label = nullptr;
    std::shared_ptr<ParallelLoopExecutor> m_executor = nullptr;
    std::shared_ptr<EmitABIRegSpills> m_parallel_section_reg_spiller = nullptr;
    mutable std::vector<uint8_t> m_common_registers_buffer;
};

class jit_parallel_loop_end_emitter : public jit_loop_end_base_emitter {
public:
    jit_parallel_loop_end_emitter(dnnl::impl::cpu::aarch64::jit_generator_t* h,
                                  dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                                  const ov::snippets::lowered::ExpressionPtr& expr);

protected:
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    std::shared_ptr<EmitABIRegSpills> m_parallel_section_reg_spiller = nullptr;
};

}  // namespace ov::intel_cpu::aarch64
