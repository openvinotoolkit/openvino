// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_loop_emitters.hpp"

#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <vector>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "jit_loop_base_emitters.hpp"
#include "openvino/core/type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {

jit_loop_begin_emitter::jit_loop_begin_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                                               dnnl::impl::cpu::x64::cpu_isa_t isa,
                                               const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa),
      jit_loop_begin_base_emitter(h, isa, expr, false) {
    auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(get_loop_end_expr(expr)->get_node());
    m_work_amount = loop_end->get_work_amount();
}

void jit_loop_begin_emitter::emit_impl([[maybe_unused]] const std::vector<size_t>& in,
                                       const std::vector<size_t>& out) const {
    // If the loop evaulate once, we can skip loop begin code emission
    // If work_amount is dynamic, we should get runtime `work_amount` - it might be `zero` and we should skip loop
    // evaluation
    const bool is_work_amount_dynamic = ov::snippets::utils::is_dynamic_value(m_work_amount);
    if (m_evaluate_once && !is_work_amount_dynamic) {
        return;
    }

    emit_loop_begin_work_amount_check(out, is_work_amount_dynamic, m_work_amount);
    h->L(*m_loop_begin_label);
}

jit_loop_end_emitter::jit_loop_end_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                                           dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_loop_end_base_emitter(h, isa, expr, false) {}

void jit_loop_end_emitter::emit_impl(const std::vector<size_t>& in,
                                     [[maybe_unused]] const std::vector<size_t>& out) const {
    emit_loop_end_impl(in, true);
    h->L(*m_loop_end_label);
}

}  // namespace ov::intel_cpu
