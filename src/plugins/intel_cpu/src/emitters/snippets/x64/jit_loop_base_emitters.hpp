// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xbyak/xbyak.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "cpu/x64/jit_generator.hpp"
#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu {

class jit_loop_begin_base_emitter : public virtual jit_emitter {
public:
    jit_loop_begin_base_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                                dnnl::impl::cpu::x64::cpu_isa_t isa,
                                const ov::snippets::lowered::ExpressionPtr& expr,
                                bool is_parallel);

    size_t get_inputs_num() const override {
        return 0;
    }

    // `jit_loop_begin_emitter` handles manually aux_gpr allocation using `jit_aux_gpr_holder`
    size_t aux_gprs_count() const override {
        return 0;
    }

    void set_loop_end_label(const std::shared_ptr<const Xbyak::Label>& label) {
        m_loop_end_label = label;
    }

    std::shared_ptr<const Xbyak::Label> get_begin_label() const {
        return m_loop_begin_label;
    }

protected:
    std::shared_ptr<Xbyak::Label> m_loop_begin_label = nullptr;
    std::shared_ptr<const Xbyak::Label> m_loop_end_label = nullptr;
    size_t m_wa_increment = 0;
    size_t m_loop_id_offset = 0;
    bool m_evaluate_once = false;

    static ov::snippets::lowered::ExpressionPtr get_loop_end_expr(const ov::snippets::lowered::ExpressionPtr& expr);

    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    /**
     * @brief Loads work amount (either from runtime arguments for dynamic loops
     * or uses a compile-time constant for static loops), and jumps to
     * the loop end label if the work amount is less than the increment, effectively skipping the loop body.
     *
     * @param out Vector of output register indices, contining the work amount register
     * @param is_work_amount_dynamic True if work amount is determined at runtime
     * (and its value should be read from loop_args), false if known at compile time
     * @param work_amount_static Compile-time work amount value (used only when is_work_amount_dynamic is false)
     */
    void emit_loop_begin_work_amount_check(const std::vector<size_t>& out,
                                           bool is_work_amount_dynamic,
                                           int64_t work_amount_static) const;

    void emit_code_impl(const std::vector<size_t>& in_idxs,
                        const std::vector<size_t>& out_idxs,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override;
};

class jit_loop_end_base_emitter : public jit_emitter {
public:
    jit_loop_end_base_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                              dnnl::impl::cpu::x64::cpu_isa_t isa,
                              const ov::snippets::lowered::ExpressionPtr& expr,
                              bool is_parallel);

    size_t get_inputs_num() const override {
        return 0;
    }

    static jit_snippets_call_args::loop_args_t compose_loop_args(
        const std::shared_ptr<ov::snippets::op::LoopEnd>& loop_end);

    // `jit_loop_end_base_emitter` handles manually aux_gpr allocation using `jit_aux_gpr_holder`
    size_t aux_gprs_count() const override {
        return 0;
    }

protected:
    void emit_code_impl(const std::vector<size_t>& in_idxs,
                        const std::vector<size_t>& out_idxs,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override;

    static ov::snippets::lowered::ExpressionPtr get_loop_begin_expr(const ov::snippets::lowered::ExpressionPtr& expr);

    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    /**
     * @brief Emits loop termination logic.
     * If evaluate once is true, applies only finalization offsets if needed.
     * Otherwise:
     * 1. Applies pointer increments to advance data pointers for the next iteration
     * 2. Decrements the work amount by the loop increment
     * 3. Checks if remaining work amount >= increment and jumps back to loop begin if true
     *
     * @param in Vector of reg indices, where data pointer registers come first and work amount register is last
     * @param apply_finalization_offsets If true, applies finalization offsets to data ptrs
     */
    void emit_loop_end_logic(const std::vector<size_t>& in, bool apply_finalization_offsets) const;

    std::shared_ptr<Xbyak::Label> m_loop_end_label = nullptr;
    std::shared_ptr<const Xbyak::Label> m_loop_begin_label = nullptr;
    size_t m_io_num = 0;
    size_t m_wa_increment = 0;
    size_t m_loop_id_offset = 0;
    bool m_evaluate_once = false;
    bool m_are_ptr_increments_dynamic = false;
    bool m_are_final_offsets_dynamic = false;
    jit_snippets_call_args::loop_args_t m_loop_args;

private:
    void apply_increments_to_ptrs(const std::vector<size_t>& data_ptr_reg_idxs,
                                  const int64_t* increments,
                                  bool use_runtime_args,
                                  size_t field_offset,
                                  const std::vector<size_t>& used_aux_gprs) const;
};

}  // namespace ov::intel_cpu
