// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "emitters/plugin/x64/jit_emitter.hpp"

#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace intel_cpu {

/* ================== jit_loop_begin_emitter ====================== */

class jit_loop_begin_emitter: public jit_emitter {
public:
    jit_loop_begin_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                           const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override { return 0; }

    void emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                  const std::vector<size_t> &pool_vec_idxs = {}, const std::vector<size_t> &pool_gpr_idxs = {}) const override;

    void set_loop_end_label(const std::shared_ptr<const Xbyak::Label>& label) { loop_end_label = label; }
    std::shared_ptr<const Xbyak::Label> get_begin_label() { return loop_begin_label; }

protected:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    // `jit_loop_begin_emitter` handles manually aux_gpr allocation using `jit_aux_gpr_holder`
    size_t aux_gprs_count() const override { return 0; }

    std::shared_ptr<Xbyak::Label> loop_begin_label = nullptr;
    std::shared_ptr<const Xbyak::Label> loop_end_label = nullptr;
    size_t work_amount = 0;
    size_t wa_increment = 0;
    size_t loop_id = 0;
    bool evaluate_once = false;
    bool is_work_amount_dynamic = false;
};


/* ============================================================== */

/* ================== jit_loop_end_emitter ====================== */

class jit_loop_end_emitter: public jit_emitter {
public:
    jit_loop_end_emitter(dnnl::impl::cpu::x64::jit_generator* h, dnnl::impl::cpu::x64::cpu_isa_t isa,
                           const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override { return 0; }

    void emit_code(const std::vector<size_t> &in_idxs, const std::vector<size_t> &out_idxs,
                   const std::vector<size_t> &pool_vec_idxs = {}, const std::vector<size_t> &pool_gpr_idxs = {}) const override;

protected:
    void validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    // `jit_loop_end_emitter` handles manually aux_gpr allocation using `jit_aux_gpr_holder`
    size_t aux_gprs_count() const override { return 0; }

    static ov::snippets::lowered::ExpressionPtr get_loop_begin_expr(const ov::snippets::lowered::ExpressionPtr& expr);

    std::shared_ptr<const Xbyak::Label> loop_begin_label = nullptr;
    std::shared_ptr<Xbyak::Label> loop_end_label = nullptr;
    size_t num_inputs = 0;
    size_t num_outputs = 0;
    size_t work_amount = 0;
    size_t wa_increment = 0;
    std::vector<bool> is_incremented = {};
    std::vector<int64_t> ptr_increments = {};
    std::vector<int64_t> finalization_offsets = {};
    std::vector<int64_t> data_sizes = {};
    size_t loop_id = 0;
    bool evaluate_once = false;
    bool are_ptr_increments_dynamic = false;
    bool are_final_offsets_dynamic = false;
    bool are_ptr_shifts_dynamic = false;
};

/* ============================================================== */

}   // namespace intel_cpu
}   // namespace ov
