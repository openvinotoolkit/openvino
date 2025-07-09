// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "emitters/plugin/aarch64/jit_emitter.hpp"
#include "snippets/lowered/expression.hpp"

namespace ov::intel_cpu::aarch64 {

/* ================== jit_loop_begin_emitter ====================== */

class jit_loop_begin_emitter : public jit_emitter {
public:
    jit_loop_begin_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                           dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                           const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {
        return 0;
    }
    std::shared_ptr<const Xbyak_aarch64::Label> get_begin_label() {
        return loop_begin_label;
    }

    void set_loop_end_label(const std::shared_ptr<Xbyak_aarch64::Label>& label) {
        loop_end_label = label;
    }

protected:
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    void emit_code_impl(const std::vector<size_t>& in_idxs,
                        const std::vector<size_t>& out_idxs,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override;

    std::shared_ptr<Xbyak_aarch64::Label> loop_begin_label;
    std::shared_ptr<Xbyak_aarch64::Label> loop_end_label;
    size_t work_amount = 0;
    size_t wa_increment = 0;
    bool evaluate_once = false;
    size_t loop_id = 0;
    bool is_work_amount_dynamic = false;
};

/* ============================================================== */

/* ================== jit_loop_end_emitter ====================== */

class jit_loop_end_emitter : public jit_emitter {
public:
    jit_loop_end_emitter(dnnl::impl::cpu::aarch64::jit_generator* h,
                         dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                         const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override {
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

    std::shared_ptr<const Xbyak_aarch64::Label> loop_begin_label;
    std::shared_ptr<Xbyak_aarch64::Label> loop_end_label;
    size_t num_inputs = 0;
    size_t num_outputs = 0;
    size_t work_amount = 0;
    size_t wa_increment = 0;
    std::vector<bool> is_incremented;
    std::vector<int64_t> ptr_increments;
    std::vector<int64_t> finalization_offsets;
    std::vector<int64_t> data_sizes;
    bool evaluate_once = false;
    size_t loop_id = 0;
};

/* ============================================================== */

}  // namespace ov::intel_cpu::aarch64
