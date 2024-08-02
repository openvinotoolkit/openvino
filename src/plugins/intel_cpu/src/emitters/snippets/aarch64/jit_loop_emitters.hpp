// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/aarch64/jit_emitter.hpp"

#include "snippets/op/loop.hpp"

namespace ov {
namespace intel_cpu {
namespace aarch64 {

/* ================== jit_loop_begin_emitter ====================== */

class jit_loop_begin_emitter: public jit_emitter {
public:
    jit_loop_begin_emitter(dnnl::impl::cpu::aarch64::jit_generator* h, dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                           const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override { return 0; }

    void emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                   const std::vector<size_t> &pool_vec_idxs = {}, const std::vector<size_t> &pool_gpr_idxs = {}) const override;

    std::shared_ptr<const Xbyak_aarch64::Label> get_begin_label() { return loop_begin_label; }

protected:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    std::shared_ptr<Xbyak_aarch64::Label> loop_begin_label;
    size_t work_amount = 0;
    int64_t wa_increment = 0;
    bool evaluate_once = false;
};

/* ============================================================== */

/* ================== jit_loop_end_emitter ====================== */

class jit_loop_end_emitter: public jit_emitter {
public:
    jit_loop_end_emitter(dnnl::impl::cpu::aarch64::jit_generator* h, dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                           const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_count() const override { return 0; }

    void emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                   const std::vector<size_t> &pool_vec_idxs = {}, const std::vector<size_t> &pool_gpr_idxs = {}) const override;

protected:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    static ov::snippets::lowered::ExpressionPtr get_loop_begin_expr(const ov::snippets::lowered::ExpressionPtr& expr);

    std::shared_ptr<const Xbyak_aarch64::Label> loop_begin_label;
    size_t num_inputs = 0;
    size_t num_outputs = 0;
    size_t work_amount = 0;
    int64_t wa_increment = 0;
    std::vector<bool> is_incremented = {};
    std::vector<int64_t> ptr_increments = {};
    std::vector<int64_t> finalization_offsets = {};
    std::vector<int64_t> data_sizes = {};
    bool evaluate_once = false;
};

/* ============================================================== */

}   // namespace aarch64
}   // namespace intel_cpu
}   // namespace ov
