// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <nodes/kernels/riscv64/jit_generator.hpp>
#include <nodes/kernels/riscv64/cpu_isa_traits.hpp>
#include "emitters/plugin/riscv64/jit_emitter.hpp"
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

    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const;
    void emit_code_impl(const std::vector<size_t>& in,
                        const std::vector<size_t>& out,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override;

    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const;

    Xbyak_riscv::Label get_begin_label() const { return loop_begin_label; }
    void set_loop_end_label(const Xbyak_riscv::Label& lbl) { this->loop_end_label = const_cast<Xbyak_riscv::Label*>(&lbl); }

private:
    ov::intel_cpu::riscv64::cpu_isa_t isa;
    ov::intel_cpu::riscv64::jit_generator_t* h;
    bool evaluate_once = false;
    size_t work_amount = 0lu;
    int64_t increment = 0;
    bool is_work_amount_dynamic = false;
    mutable Xbyak_riscv::Label loop_begin_label;
    mutable Xbyak_riscv::Label* loop_end_label = nullptr;
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

    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const;
    void emit_code_impl(const std::vector<size_t>& in,
                        const std::vector<size_t>& out,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override;

    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const;

private:
    static ov::snippets::lowered::ExpressionPtr get_loop_begin_expr(const ov::snippets::lowered::ExpressionPtr& expr);

    ov::intel_cpu::riscv64::cpu_isa_t isa;
    ov::intel_cpu::riscv64::jit_generator_t* h;
    size_t num_inputs = 0;
    size_t num_outputs = 0;
    int64_t work_amount = 0;
    int64_t increment = 0;
    std::vector<bool> is_incremented;
    std::vector<int64_t> ptr_increments;
    std::vector<int64_t> finalization_offsets;
    std::vector<int64_t> data_sizes;
    bool evaluate_once = false;
    bool is_increment_dynamic = false;
    bool are_ptr_increments_dynamic = false;
    bool are_final_offsets_dynamic = false;
    mutable Xbyak_riscv::Label loop_begin_label;
    mutable Xbyak_riscv::Label loop_end_label;
};

}  // namespace ov::intel_cpu::riscv64
