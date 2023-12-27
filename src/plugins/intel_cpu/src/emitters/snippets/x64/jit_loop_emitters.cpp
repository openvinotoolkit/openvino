// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_loop_emitters.hpp"


using namespace Xbyak;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {

inline static void transform_idxs_to_regs(const std::vector<size_t>& idxs, std::vector<Reg64>& regs) {
    regs.resize(idxs.size());
    std::transform(idxs.begin(), idxs.end(), regs.begin(), [](size_t idx){return Reg64(static_cast<int>(idx));});
}

jit_loop_begin_emitter::jit_loop_begin_emitter(jit_generator* h, cpu_isa_t isa, const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    loop_begin = ov::as_type_ptr<snippets::op::LoopBegin>(expr->get_node());
    OPENVINO_ASSERT(loop_begin != nullptr, "jit_loop_begin_emitter invoked with invalid op argument");
    const auto loop_end = get_loop_end(expr);
    work_amount = loop_end->get_work_amount();
    evaluate_once = loop_end->get_evaluate_once();
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

std::shared_ptr<snippets::op::LoopEnd> jit_loop_begin_emitter::get_loop_end(const ov::snippets::lowered::ExpressionPtr& expr) {
    OPENVINO_ASSERT(expr->get_output_port_connectors().size() == 1, "jit_loop_begin_emitter has invalid LoopBegin expression configuration");
    const auto& consumers = expr->get_output_port_connector(0)->get_consumers();
    OPENVINO_ASSERT(consumers.size() == 1, "jit_loop_begin_emitter has invalid LoopBegin expression configuration");
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(consumers.cbegin()->get_expr()->get_node());
    OPENVINO_ASSERT(loop_end != nullptr, "jit_loop_begin_emitter has invalid LoopBegin expression configuration");
    return loop_end;
}

void jit_loop_begin_emitter::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
                                       const std::vector<size_t> &pool_vec_idx, const std::vector<size_t> &pool_gpr_idxs) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void jit_loop_begin_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    OPENVINO_ASSERT(in.empty(), "Invalid inputs size: expected 0 got ", in.size());
    OPENVINO_ASSERT(out.size() == 1, "Invalid outputs size: expected 1 got ", out.size());
}

void jit_loop_begin_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    // todo: In dynamic case we will also need to set broadcasting info here
    Reg64 reg_work_amount = Reg64(static_cast<int>(out.back()));
    Label for_body;
    // save previous register state (if there is an outer loop that uses this reg for example)
    if (!evaluate_once) {
        h->mov(reg_work_amount, work_amount);
    }
    // Note: loop address is not calculated at this point, so need to call calcJmpAddress() which is protected
    // or ready(), but they both set internal flags and that's not a desired way to use them.
    // So the most obvious WA is just to use current address manually
    loop_begin->begin_address = h->getCurr();
}

jit_loop_end_emitter::jit_loop_end_emitter(jit_generator* h, cpu_isa_t isa, const ov::snippets::lowered::ExpressionPtr& expr)
    : jit_emitter(h, isa) {
    loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(expr->get_node());
    OPENVINO_ASSERT(loop_end != nullptr, "jit_loop_end_emitter invoked with invalid op argument");
    loop_begin = loop_end->get_loop_begin();
    // Note that 1 edge connects LoopBegin and LoopEnd
    num_inputs = expr->get_input_count();
    num_outputs = expr->get_output_count();
    wa_increment = static_cast<int64_t>(loop_end->get_increment());
    work_amount = static_cast<int64_t>(loop_end->get_work_amount());
    ptr_increments = loop_end->get_ptr_increments();
    finalization_offsets = loop_end->get_finalization_offsets();
    evaluate_once = loop_end->get_evaluate_once();
    io_data_size = loop_end->get_element_type_sizes();
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void jit_loop_end_emitter::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out,
                                     const std::vector<size_t> &pool_vec_idx, const std::vector<size_t> &pool_gpr_idxs) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}


void jit_loop_end_emitter::validate_arguments(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    const auto io_size = num_inputs - 1;
    OPENVINO_ASSERT(in.size() == num_inputs, "Invalid number of in arguments: expected ", num_inputs , " got ", in.size());
    OPENVINO_ASSERT(out.size() == num_outputs, "Invalid number of out arguments: expected ", num_outputs, " got ", out.size());
    OPENVINO_ASSERT(ptr_increments.size() == io_size, "Invalid ptr_increments size: expected ", io_size, " got ", ptr_increments.size());
    OPENVINO_ASSERT(finalization_offsets.size() == io_size, "Invalid finalization_offsets size: expected ", io_size, " got ", finalization_offsets.size());
}

void jit_loop_end_emitter::emit_impl(const std::vector<size_t>& in, const std::vector<size_t>& out) const {
    std::vector<size_t> data_ptr_reg_idxs;
    // the last input is actually a work_amount reg
    data_ptr_reg_idxs.reserve(num_inputs - 1);
    std::copy(in.begin(), in.end() - 1, std::back_inserter(data_ptr_reg_idxs));
    std::vector<Reg64> data_ptr_regs;
    transform_idxs_to_regs(data_ptr_reg_idxs, data_ptr_regs);
    Reg64 reg_work_amount = Reg64(in.back());
    if (!evaluate_once) {
        for (size_t idx = 0; idx < data_ptr_regs.size(); idx++) {
            if (ptr_increments[idx] != 0)
                h->add(data_ptr_regs[idx], ptr_increments[idx] * wa_increment * io_data_size[idx]);
        }
        h->sub(reg_work_amount, wa_increment);
        h->cmp(reg_work_amount, wa_increment);
        h->jge(loop_begin->begin_address);
    }

    for (size_t idx = 0; idx < data_ptr_regs.size(); idx++) {
        if (finalization_offsets[idx] != 0)
            h->add(data_ptr_regs[idx], finalization_offsets[idx] * io_data_size[idx]);
    }
}

}   // namespace intel_cpu
}   // namespace ov
