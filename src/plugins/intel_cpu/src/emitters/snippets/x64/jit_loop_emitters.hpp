// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"

#include "snippets/op/loop.hpp"

namespace ov {
namespace intel_cpu {

/* ================== jit_loop_begin_emitter ====================== */

class jit_loop_begin_emitter: public jit_emitter {
public:
    jit_loop_begin_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                           const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override { return 0; }

    std::shared_ptr<const Xbyak::Label> get_begin_label() { return loop_begin_label; }

protected:
    static std::shared_ptr<ov::snippets::op::LoopEnd> get_loop_end(const ov::snippets::lowered::ExpressionPtr& expr);

    std::shared_ptr<Xbyak::Label> loop_begin_label;
    int64_t wa_increment = 0;
};

class jit_loop_begin_static_emitter: public jit_loop_begin_emitter {
public:
    jit_loop_begin_static_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                  const ov::snippets::lowered::ExpressionPtr& expr);

    void emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                   const std::vector<size_t> &pool_vec_idxs = {}, const std::vector<size_t> &pool_gpr_idxs = {}) const override;
protected:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    bool evaluate_once = false;
    size_t work_amount = 0;
};

class jit_loop_begin_dynamic_emitter: public jit_loop_begin_emitter {
public:
    jit_loop_begin_dynamic_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                   const ov::snippets::lowered::ExpressionPtr& expr);

    void emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                   const std::vector<size_t> &pool_vec_idxs = {}, const std::vector<size_t> &pool_gpr_idxs = {}) const override;

    void set_loop_end_label(const std::shared_ptr<const Xbyak::Label>& label) { loop_end_label = label; }

protected:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    // For Loop arguments
    size_t aux_gprs_count() const override { return 1; }

    std::shared_ptr<const Xbyak::Label> loop_end_label;
    size_t loop_id;
};

/* ============================================================== */

/* ================== jit_loop_end_emitter ====================== */

class jit_loop_end_emitter: public jit_emitter {
public:
    jit_loop_end_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                           const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override { return 0; }

protected:
    static ov::snippets::lowered::ExpressionPtr get_loop_begin_expr(const ov::snippets::lowered::ExpressionPtr& expr);

    std::shared_ptr<const Xbyak::Label> loop_begin_label;
    size_t num_inputs = 0;
    size_t num_outputs = 0;
    int64_t wa_increment = 0;
    std::vector<bool> is_incremented = {};
};

class jit_loop_end_static_emitter: public jit_loop_end_emitter {
public:
    jit_loop_end_static_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                const ov::snippets::lowered::ExpressionPtr& expr);

    void emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                   const std::vector<size_t> &pool_vec_idxs = {}, const std::vector<size_t> &pool_gpr_idxs = {}) const override;

protected:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    size_t work_amount = 0;
    std::vector<bool> is_incremented = {};
    std::vector<int64_t> ptr_increments = {};
    std::vector<int64_t> finalization_offsets = {};
    std::vector<int64_t> data_sizes = {};
    bool evaluate_once = false;
};

class jit_loop_end_dynamic_emitter: public jit_loop_end_emitter {
public:
    jit_loop_end_dynamic_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                                 const ov::snippets::lowered::ExpressionPtr& expr);

    void emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                   const std::vector<size_t> &pool_vec_idxs = {}, const std::vector<size_t> &pool_gpr_idxs = {}) const override;

protected:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    // For Loop arguments
    size_t aux_gprs_count() const override { return 1; }

    std::shared_ptr<Xbyak::Label> loop_end_label;
    size_t loop_id;
};

/* ============================================================== */

}   // namespace intel_cpu
}   // namespace ov
