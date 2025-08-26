// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xbyak/xbyak.h>

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "jit_loop_base_emitters.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::intel_cpu {
struct loop_begin_common_fields {
    std::shared_ptr<Xbyak::Label> loop_begin_label = nullptr;
    std::shared_ptr<const Xbyak::Label> loop_end_label = nullptr;
    size_t wa_increment = 0;
    size_t loop_id_offset = 0;
    bool evaluate_once = false;
    std::shared_ptr<ov::snippets::op::LoopEnd> loop_end = nullptr;

    void init_from_expr(const ov::snippets::lowered::ExpressionPtr& expr) {
        const auto loop_begin = ov::as_type_ptr<snippets::op::LoopBegin>(expr->get_node());
        OV_CPU_JIT_EMITTER_ASSERT(loop_begin, "expects LoopBegin expression");
        loop_end = loop_begin->get_loop_end();
        wa_increment = loop_end->get_increment();
        evaluate_once = loop_end->get_evaluate_once();
        loop_id_offset = loop_end->get_id() * sizeof(jit_snippets_call_args::loop_args_t);
    }

    void validate_loop_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
        OV_CPU_JIT_EMITTER_ASSERT(in.empty(), "Invalid inputs size: expected 0 got " + std::to_string(in.size()));
        // Note: the only expected output is work amount register (communicated to jit_loop_end_emitter)
        OV_CPU_JIT_EMITTER_ASSERT(out.size() == 1, "Invalid outputs size: expected 1 got " + std::to_string(out.size()));
        OV_CPU_JIT_EMITTER_ASSERT(loop_begin_label != nullptr && loop_end_label != nullptr, "has not inited labels!");
        OV_CPU_JIT_EMITTER_ASSERT(!ov::snippets::utils::is_dynamic_value(wa_increment) || evaluate_once,
                                  "loop increment might be dynamic only if loop evaluates once!");
    }
};

class jit_loop_begin_emitter : public jit_emitter {
public:
    jit_loop_begin_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                           dnnl::impl::cpu::x64::cpu_isa_t isa,
                           const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {
        return 0;
    }

    void set_loop_end_label(const std::shared_ptr<const Xbyak::Label>& label) {
        common_fields.loop_end_label = label;
    }
    std::shared_ptr<const Xbyak::Label> get_begin_label() const {
        return common_fields.loop_begin_label;
    }

    // `jit_loop_begin_emitter` handles manually aux_gpr allocation using `jit_aux_gpr_holder`
    size_t aux_gprs_count() const override {
        return 0;
    }

protected:
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    void emit_code_impl(const std::vector<size_t>& in_idxs,
                        const std::vector<size_t>& out_idxs,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override;

    loop_begin_common_fields common_fields;
    size_t work_amount = 0;
    size_t loop_id = 0;
    bool is_work_amount_dynamic = false;
};

class jit_loop_end_emitter : public jit_loop_end_base_emitter {
public:
    jit_loop_end_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                         dnnl::impl::cpu::x64::cpu_isa_t isa,
                         const ov::snippets::lowered::ExpressionPtr& expr);

protected:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
};

}  // namespace ov::intel_cpu
