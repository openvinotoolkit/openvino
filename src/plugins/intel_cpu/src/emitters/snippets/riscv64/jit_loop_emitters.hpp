// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <nodes/kernels/riscv64/cpu_isa_traits.hpp>
#include <nodes/kernels/riscv64/jit_generator.hpp>

#include "emitters/plugin/riscv64/jit_emitter.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "snippets/lowered/expression.hpp"

namespace ov::intel_cpu::riscv64 {

/* ================== jit_loop_begin_emitter ====================== */

class jit_loop_begin_emitter : public ov::intel_cpu::riscv64::jit_emitter {
public:
    jit_loop_begin_emitter(ov::intel_cpu::riscv64::jit_generator_t* h,
                           ov::intel_cpu::riscv64::cpu_isa_t isa,
                           const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {
        return 0;
    }

    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_code_impl(const std::vector<size_t>& in,
                        const std::vector<size_t>& out,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override;

    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    std::shared_ptr<const Xbyak_riscv::Label> get_begin_label() const {
        return loop_begin_label;
    }
    void set_loop_end_label(const std::shared_ptr<const Xbyak_riscv::Label>& lbl) {
        this->loop_end_label = lbl;
    }

private:
    ov::intel_cpu::riscv64::cpu_isa_t isa;
    ov::intel_cpu::riscv64::jit_generator_t* h;
    bool evaluate_once = false;
    size_t work_amount = 0LU;
    size_t wa_increment = 0;
    size_t loop_id = 0;
    bool is_work_amount_dynamic = false;
    mutable std::shared_ptr<Xbyak_riscv::Label> loop_begin_label = nullptr;
    mutable std::shared_ptr<const Xbyak_riscv::Label> loop_end_label = nullptr;
};

/* =================== jit_loop_end_emitter ======================= */

class jit_loop_end_emitter : public ov::intel_cpu::riscv64::jit_emitter {
public:
    jit_loop_end_emitter(ov::intel_cpu::riscv64::jit_generator_t* h,
                         ov::intel_cpu::riscv64::cpu_isa_t isa,
                         const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {
        return 0;
    }

    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_code_impl(const std::vector<size_t>& in,
                        const std::vector<size_t>& out,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override;

    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

private:
    static ov::snippets::lowered::ExpressionPtr get_loop_begin_expr(const ov::snippets::lowered::ExpressionPtr& expr);

    ov::intel_cpu::riscv64::cpu_isa_t isa;
    ov::intel_cpu::riscv64::jit_generator_t* h;
    size_t num_inputs = 0;
    size_t num_outputs = 0;
    int64_t work_amount = 0;
    size_t wa_increment = 0;
    size_t loop_id = 0;
    bool evaluate_once = false;
    bool are_ptr_increments_dynamic = false;
    bool are_final_offsets_dynamic = false;
    size_t loop_args_offset = 0;
    jit_snippets_call_args::loop_args_t loop_args;
    mutable std::shared_ptr<const Xbyak_riscv::Label> loop_begin_label = nullptr;
    mutable std::shared_ptr<Xbyak_riscv::Label> loop_end_label = nullptr;
    mutable bool end_label_bound = false;
};

}  // namespace ov::intel_cpu::riscv64
