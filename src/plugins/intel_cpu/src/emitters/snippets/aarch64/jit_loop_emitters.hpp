// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <vector>

#include "jit_loop_base_emitters.hpp"
#include "snippets/lowered/expression.hpp"

namespace ov::intel_cpu::aarch64 {

/* ================== jit_loop_begin_emitter ====================== */

class jit_loop_begin_emitter : public jit_loop_begin_base_emitter {
public:
    jit_loop_begin_emitter(dnnl::impl::cpu::aarch64::jit_generator_t* h,
                           dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                           const ov::snippets::lowered::ExpressionPtr& expr);

    size_t get_aux_gprs_count() const override {
        return 0;
    }

protected:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;

    size_t m_work_amount = 0;
};

/* ============================================================== */

/* ================== jit_loop_end_emitter ====================== */

class jit_loop_end_emitter : public jit_loop_end_base_emitter {
public:
    jit_loop_end_emitter(dnnl::impl::cpu::aarch64::jit_generator_t* h,
                         dnnl::impl::cpu::aarch64::cpu_isa_t isa,
                         const ov::snippets::lowered::ExpressionPtr& expr);

protected:
    void emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const override;
};

/* ============================================================== */

}  // namespace ov::intel_cpu::aarch64
