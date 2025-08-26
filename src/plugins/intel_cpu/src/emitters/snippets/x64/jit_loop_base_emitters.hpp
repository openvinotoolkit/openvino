// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xbyak/xbyak.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/loop.hpp"

namespace ov::intel_cpu {
class jit_loop_end_base_emitter : public jit_emitter {
public:
    jit_loop_end_base_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
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

    // `jit_loop_end_base_emitter` handles manually aux_gpr allocation using `jit_aux_gpr_holder`
    size_t aux_gprs_count() const override {
        return 0;
    }

    static ov::snippets::lowered::ExpressionPtr get_loop_begin_expr(const ov::snippets::lowered::ExpressionPtr& expr);

    static jit_snippets_call_args::loop_args_t compose_loop_args(
        const std::shared_ptr<ov::snippets::op::LoopEnd>& loop_end);

    void apply_increments_to_ptrs(const std::vector<size_t>& data_ptr_reg_idxs,
                                  const int64_t* increments,
                                  bool use_runtime_args,
                                  size_t field_offset,
                                  const std::vector<size_t>& used_aux_gprs) const;

    void emit_loop_end_logic(const std::vector<size_t>& in, bool apply_finalization_offsets) const;

    std::shared_ptr<const Xbyak::Label> loop_begin_label = nullptr;
    std::shared_ptr<Xbyak::Label> loop_end_label = nullptr;
    size_t io_num = 0;
    size_t work_amount = 0;
    size_t wa_increment = 0;
    size_t loop_id_offset = 0;
    bool evaluate_once = false;
    bool are_ptr_increments_dynamic = false;
    bool are_final_offsets_dynamic = false;
    jit_snippets_call_args::loop_args_t loop_args;
};

}  // namespace ov::intel_cpu
