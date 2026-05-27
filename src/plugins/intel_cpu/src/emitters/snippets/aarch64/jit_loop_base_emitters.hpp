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
#include <vector>

#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "snippets/lowered/expression.hpp"

namespace ov::intel_cpu::aarch64 {

class jit_loop_begin_base_emitter : public virtual jit_emitter {
public:
    jit_loop_begin_base_emitter(dnnl::impl::cpu::aarch64::jit_generator_t* h,
                                dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                                const ov::snippets::lowered::ExpressionPtr& expr,
                                bool is_parallel);

    size_t get_inputs_count() const override {
        return 0;
    }

    void set_loop_end_label(const std::shared_ptr<Xbyak_aarch64::Label>& label) {
        m_loop_end_label = label;
    }

    std::shared_ptr<const Xbyak_aarch64::Label> get_begin_label() const {
        return m_loop_begin_label;
    }

protected:
    std::shared_ptr<Xbyak_aarch64::Label> m_loop_begin_label = nullptr;
    std::shared_ptr<Xbyak_aarch64::Label> m_loop_end_label = nullptr;
    size_t m_wa_increment = 0;
    size_t m_loop_id_offset = 0;
    bool m_evaluate_once = false;

    static ov::snippets::lowered::ExpressionPtr get_loop_end_expr(const ov::snippets::lowered::ExpressionPtr& expr);

    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
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
    jit_loop_end_base_emitter(dnnl::impl::cpu::aarch64::jit_generator_t* h,
                              dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                              const ov::snippets::lowered::ExpressionPtr& expr,
                              bool is_parallel);

    size_t get_inputs_count() const override {
        return 0;
    }

    size_t get_aux_gprs_count() const override {
        return 0;
    }

protected:
    void emit_code_impl(const std::vector<size_t>& in_idxs,
                        const std::vector<size_t>& out_idxs,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override;

    static ov::snippets::lowered::ExpressionPtr get_loop_begin_expr(const ov::snippets::lowered::ExpressionPtr& expr);

    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_loop_end_impl(const std::vector<size_t>& in, bool apply_finalization_offsets) const;

    std::shared_ptr<Xbyak_aarch64::Label> m_loop_end_label = nullptr;
    std::shared_ptr<const Xbyak_aarch64::Label> m_loop_begin_label = nullptr;
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
                                  size_t field_offset) const;
};

}  // namespace ov::intel_cpu::aarch64
