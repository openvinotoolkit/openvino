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
    const auto& target_inputs = loop_begin->output(loop_begin->get_output_size() - 1).get_target_inputs();
    // todo: this check could be excessive, since we check for it in validate_and_infer_types()
    if (target_inputs.size() != 1)
        OPENVINO_THROW("jit_loop_begin_emitter invoked with invalid configuration: the last output must have exactly one "
                       "input attached");
    const auto loop_end = ov::as_type_ptr<snippets::op::LoopEnd>(target_inputs.begin()->get_node()->shared_from_this());
    if (!loop_end)
        OPENVINO_THROW("jit_loop_begin_emitter invoked with invalid configuration: the last output must be LoopEnd");
    work_amount = loop_end->get_work_amount();
    evaluate_once = loop_end->get_evaluate_once();
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void jit_loop_begin_emitter::emit_code(const std::vector<size_t> &in, const std::vector<size_t> &out) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}

void jit_loop_begin_emitter::validate_arguments(const std::vector<size_t> &in,
                                          const std::vector<size_t> &out) const {
    if (!in.empty())
        OPENVINO_THROW("Invalid inputs size: expected 0 got ", in.size());
    if (out.size() != 1)
        OPENVINO_THROW("Invalid outputs size: expected 1 got ", out.size());
}

void jit_loop_begin_emitter::emit_impl(const std::vector<size_t>& in,
                                 const std::vector<size_t>& out) const {
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
    if (!loop_end)
        OPENVINO_THROW("jit_loop_end_emitter invoked with invalid op argument");
    loop_begin = loop_end->get_loop_begin();
    // todo: this check could be excessive, since we check for it in validate_and_infer_types()
    if (!loop_begin)
        OPENVINO_THROW("jit_loop_end_emitter invoked with invalid configuration: the last arg must be LoopBegin");
    // Note that 1 edge connects LoopBegin and LoopEnd
    num_inputs = loop_end->get_input_num();
    num_outputs = loop_end->get_output_num();
    wa_increment = static_cast<int64_t>(loop_end->get_increment());
    work_amount = static_cast<int64_t>(loop_end->get_work_amount());
    ptr_increments = loop_end->get_ptr_increments();
    finalization_offsets = loop_end->get_finalization_offsets();
    evaluate_once = loop_end->get_evaluate_once();
    io_data_size = loop_end->get_element_type_sizes();
    in_out_type_ = emitter_in_out_map::gpr_to_gpr;
}

void jit_loop_end_emitter::emit_code(const std::vector<size_t> &in,
                                 const std::vector<size_t> &out) const {
    validate_arguments(in, out);
    emit_impl(in, out);
}


void jit_loop_end_emitter::validate_arguments(const std::vector<size_t> &in,
                                        const std::vector<size_t> &out) const {
    if (out.size() != num_outputs)
        OPENVINO_THROW("Invalid number of out arguments: expected ", num_outputs, " got ", out.size());
    if (in.size() != num_inputs)
        OPENVINO_THROW("Invalid number of in arguments: expected ", num_inputs , " got ", in.size());
    const auto io_size = num_inputs - 1;
    if (ptr_increments.size() != io_size)
        OPENVINO_THROW("Invalid ptr_increments size: expected ", io_size, " got ", ptr_increments.size());
    if (finalization_offsets.size() != io_size)
        OPENVINO_THROW("Invalid finalization_offsets size: expected: ", io_size, " got ", finalization_offsets.size());
}

void jit_loop_end_emitter::emit_impl(const std::vector<size_t>& in,
                                 const std::vector<size_t>& out) const {
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
