// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/plugin/x64/utils.hpp"
#include "emitters/snippets/x64/kernel_executors/parallel_loop.hpp"
#include "jit_binary_call_emitter.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu {

class jit_parallel_loop_base_emitter : public jit_binary_call_emitter {
public:
    jit_parallel_loop_base_emitter(dnnl::impl::cpu::x64::jit_generator* h,
                                   dnnl::impl::cpu::x64::cpu_isa_t isa,
                                   const ov::snippets::lowered::ExpressionPtr& expr);

protected:
    size_t wa_increment = 0;
    std::vector<bool> is_incremented;
    size_t num_inputs = 0;
    size_t num_outputs = 0;
    size_t loop_id_offset = 0;
    bool evaluate_once = false;
    bool is_dynamic = false;
    size_t internal_work_amount_reg_idx = 0;
    std::vector<size_t> mem_ptr_regs_idxs;
    jit_snippets_call_args::loop_args_t loop_args;
};

/* ================== jit_loop_begin_emitter ====================== */
class jit_parallel_loop_begin_emitter : public jit_parallel_loop_base_emitter {
public:
    jit_parallel_loop_begin_emitter(dnnl::impl::cpu::x64::jit_generator* h,
                                    dnnl::impl::cpu::x64::cpu_isa_t isa,
                                    const ov::snippets::lowered::ExpressionPtr& expr,
                                    const snippets::KernelExecutorTablePtr& kernel_table);

    size_t get_inputs_num() const override {
        return 0;
    }

    void set_loop_end_label(const std::shared_ptr<const Xbyak::Label>& label) {
        loop_end_label = label;
    }
    std::shared_ptr<const Xbyak::Label> get_begin_label() {
        return loop_begin_label;
    }

    std::shared_ptr<EmitABIRegSpills> get_seq_part_spiller() {
        return m_seq_part_spiller;
    }

    std::shared_ptr<EmitABIRegSpills> get_par_to_seq_part_spiller() {
        return m_par_to_seq_part_spiller;
    }

protected:
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    void emit_parallel_executor_call() const;
    void emit_parallel_region_initialization() const;
    std::set<snippets::Reg> get_regs_to_spill_except_mem_ptr_regs() const;

    void emit_code_impl(const std::vector<size_t>& in_idxs,
                        const std::vector<size_t>& out_idxs,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override;

    std::shared_ptr<Xbyak::Label> loop_begin_label = nullptr;
    std::shared_ptr<Xbyak::Label> loop_preamble_label = nullptr;
    std::shared_ptr<const Xbyak::Label> loop_end_label = nullptr;
    std::shared_ptr<ParallelLoopExecutor> m_executor = nullptr;
    std::shared_ptr<EmitABIRegSpills> m_seq_part_spiller = nullptr;
    std::shared_ptr<EmitABIRegSpills> m_par_to_seq_part_spiller = nullptr;
};

class jit_parallel_loop_end_emitter : public jit_parallel_loop_base_emitter {
public:
    jit_parallel_loop_end_emitter(dnnl::impl::cpu::x64::jit_generator* h,
                                  dnnl::impl::cpu::x64::cpu_isa_t isa,
                                  const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {
        return 0;
    }

    void emit_code_impl(const std::vector<size_t>& in_idxs,
                        const std::vector<size_t>& out_idxs,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override;

protected:
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    static ov::snippets::lowered::ExpressionPtr get_loop_begin_expr(const ov::snippets::lowered::ExpressionPtr& expr);

    std::shared_ptr<const Xbyak::Label> loop_begin_label = nullptr;
    std::shared_ptr<Xbyak::Label> loop_end_label = nullptr;
    std::shared_ptr<EmitABIRegSpills> m_seq_part_spiller = nullptr;
    std::shared_ptr<EmitABIRegSpills> m_par_to_seq_part_spiller = nullptr;
};

}  // namespace ov::intel_cpu
