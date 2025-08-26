// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_loop_emitters.hpp"

#include <xbyak/xbyak.h>

#include <algorithm>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "emitters/plugin/x64/jit_emitter.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/x64/utils.hpp"
#include "emitters/utils.hpp"
#include "jit_loop_base_emitters.hpp"
#include "openvino/core/type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/utils/utils.hpp"

using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov::intel_cpu {

/* ================== jit_loop_begin_emitter ====================== */

jit_loop_begin_emitter::jit_loop_begin_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                                               dnnl::impl::cpu::x64::cpu_isa_t isa,
                                               const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    common_fields.loop_begin_label = std::make_shared<Xbyak::Label>();
    common_fields.init_from_expr(expr);
    
    work_amount = common_fields.loop_end->get_work_amount();
    loop_id = common_fields.loop_end->get_id();
    is_work_amount_dynamic = ov::snippets::utils::is_dynamic_value(work_amount);
    
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void jit_loop_begin_emitter::validate_arguments(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    common_fields.validate_loop_arguments(in, out);
}

void jit_loop_begin_emitter::emit_code_impl(const std::vector<size_t>& in,
                                            const std::vector<size_t>& out,
                                            const std::vector<size_t>& pool_vec_idxs,
                                            const std::vector<size_t>& pool_gpr_idxs) const {
    validate_arguments(in, out);
    jit_emitter::emit_code_impl(in, out, pool_vec_idxs, pool_gpr_idxs);
}

void jit_loop_begin_emitter::emit_impl([[maybe_unused]] const std::vector<size_t>& in,
                                       const std::vector<size_t>& out) const {
    // If the loop evaulate once, we can skip loop begin code emission
    // If work_amount is dynamic, we should get runtime `work_amount` - it might be `zero` and we should skip loop
    // evaluation
    if (common_fields.evaluate_once && !is_work_amount_dynamic) {
        return;
    }

    const auto loop_id_offset_for_runtime = loop_id * sizeof(jit_snippets_call_args::loop_args_t);
    jit_loop_end_base_emitter::emit_loop_begin_work_amount_check(
        h, aux_gpr_idxs, out, is_work_amount_dynamic, work_amount, loop_id_offset_for_runtime,
        common_fields.evaluate_once, common_fields.wa_increment, common_fields.loop_end_label);

    h->L(*common_fields.loop_begin_label);
}

jit_loop_end_emitter::jit_loop_end_emitter(dnnl::impl::cpu::x64::jit_generator_t* h,
                                           dnnl::impl::cpu::x64::cpu_isa_t isa,
                                           const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_loop_end_base_emitter(h, isa, expr) {
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(expr->get_node());

    const auto& ptr_increments = loop_end->get_ptr_increments();
    const auto& finalization_offsets = loop_end->get_finalization_offsets();

    are_ptr_increments_dynamic =
        std::any_of(ptr_increments.cbegin(), ptr_increments.cend(), ov::snippets::utils::is_dynamic_value<int64_t>);
    are_final_offsets_dynamic = std::any_of(finalization_offsets.cbegin(),
                                            finalization_offsets.cend(),
                                            ov::snippets::utils::is_dynamic_value<int64_t>);

    const auto begin_expr = get_loop_begin_expr(expr);
    const auto& loop_begin_emitter = std::dynamic_pointer_cast<jit_loop_begin_emitter>(begin_expr->get_emitter());
    OV_CPU_JIT_EMITTER_ASSERT(loop_begin_emitter, "LoopBegin expected jit_loop_begin_emitter");
    loop_begin_emitter->set_loop_end_label(loop_end_label);
    loop_begin_label = loop_begin_emitter->get_begin_label();
}

void jit_loop_end_emitter::emit_impl(const std::vector<size_t>& in,
                                     [[maybe_unused]] const std::vector<size_t>& out) const {
    emit_loop_end_logic(in, true);
    h->L(*loop_end_label);
}

}  // namespace ov::intel_cpu
