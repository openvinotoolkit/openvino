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
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/loop.hpp"

namespace ov::intel_cpu {

/* ================== jit_loop_begin_emitter ====================== */

class jit_loop_begin_emitter : public jit_emitter {
public:
    jit_loop_begin_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                           dnnl::impl::cpu::x64::cpu_isa_t isa,
                           const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_inputs_num() const override {
        return 0;
    }

    void set_loop_end_label(const std::shared_ptr<const Xbyak::Label>& label) {
        loop_end_label = label;
    }
    std::shared_ptr<const Xbyak::Label> get_begin_label() {
        return loop_begin_label;
    }

protected:
    void validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    void emit_code_impl(const std::vector<size_t>& in_idxs,
                        const std::vector<size_t>& out_idxs,
                        const std::vector<size_t>& pool_vec_idxs,
                        const std::vector<size_t>& pool_gpr_idxs) const override;

    // `jit_loop_begin_emitter` handles manually aux_gpr allocation using `jit_aux_gpr_holder`
    size_t aux_gprs_count() const override {
        return 0;
    }

    std::shared_ptr<Xbyak::Label> loop_begin_label = nullptr;
    std::shared_ptr<const Xbyak::Label> loop_end_label = nullptr;
    size_t work_amount = 0;
    size_t wa_increment = 0;
    size_t loop_id = 0;
    bool evaluate_once = false;
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
