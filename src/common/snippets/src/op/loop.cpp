// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/loop.hpp"
#include "snippets/generator.hpp"

using namespace std;
namespace ngraph {
namespace snippets {
namespace op {

LoopBase::LoopBase(const std::vector<Output<Node>> &args, size_t work_amount, size_t wa_increment)
        : Op(args), work_amount(work_amount), increment(wa_increment), evaluate_once(false) {
}

bool LoopBase::visit_attributes(AttributeVisitor &visitor) {
    visitor.on_attribute("work_amount", work_amount);
    visitor.on_attribute("increment", increment);
    return true;
}

size_t LoopBase::get_work_amount() const {
    return work_amount;
}

bool LoopBase::get_evaluate_once() const {
    return evaluate_once;
}

size_t LoopBase::get_increment() const {
    return increment;
}

LoopBegin::LoopBegin(const std::vector<Output<Node>> &args, size_t work_amount, size_t increment)
        : LoopBase(args, work_amount, increment),
        begin_address(nullptr), input_regs({}) {
    // We can only call a reduced validate_and_infer types from the constructor, since LoopEnd might not be attached
    // to the LoopBegin at this point (which is usually the case: create LoopBegin first => then attach LoopEnd to it)
    validate_and_infer_types_except_LoopEnd();
}

LoopBegin::LoopBegin(const std::vector<Output<Node>> &args)
        : LoopBase(args, 0, 0), begin_address(nullptr), input_regs({}) {
    validate_and_infer_types_except_LoopEnd();
}

std::shared_ptr<Node> LoopBegin::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::shared_ptr<LoopBegin>(new LoopBegin(inputs, work_amount, increment));
}


void LoopBegin::validate_and_infer_types_except_LoopEnd() {
    const size_t num_inputs = get_input_size();
    set_output_size(num_inputs + 1);
    // All outputs are by-passed from inputs, except for the last one - it connects LoopBegin and LoopEnd
    for (int i = 0; i < num_inputs; i++)
        get_output_descriptor(i).set_tensor_ptr(get_input_descriptor(i).get_output().get_tensor_ptr());
    set_output_type(num_inputs, element::f32, ov::PartialShape{ov::Shape{}});
}

void LoopBegin::validate_and_infer_types() {
    validate_and_infer_types_except_LoopEnd();
    const auto& last_output_inputs = output(get_output_size() - 1).get_target_inputs();
    NODE_VALIDATION_CHECK(this, last_output_inputs.size() == 1, "LoopBegin must have exactly one input attached to the last output");
    const auto& loop_end = ov::as_type_ptr<LoopEnd>(last_output_inputs.begin()->get_node()->shared_from_this());
    NODE_VALIDATION_CHECK(this, loop_end != nullptr, "LoopBegin must have LoopEnd connected to its last output");
    work_amount = loop_end->get_work_amount();
    increment = loop_end->get_increment();
}

std::shared_ptr<LoopEnd> LoopBegin::get_loop_end() {
    const auto& last_output_inputs = output(get_output_size() - 1).get_target_inputs();
    if (last_output_inputs.size() != 1)
        throw std::invalid_argument("LoopBegin has more than one inputs attached to the last output");
    const auto& loop_end = ov::as_type_ptr<LoopEnd>(last_output_inputs.begin()->get_node()->shared_from_this());
    if (!loop_end)
        throw std::invalid_argument("LoopBegin last output is not connected to LoopEnd");
    return  loop_end;
}

LoopEnd::LoopEnd(const std::vector<Output<Node>> &args, size_t work_amount, size_t increment,
                 std::vector<bool> apply_increment, std::vector<int64_t> finalization_offsets)
        : LoopBase(args, work_amount, increment), apply_increment(std::move(apply_increment)),
        finalization_offsets(std::move(finalization_offsets)), has_outer_loop(true) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> LoopEnd::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<LoopEnd>(inputs, work_amount, increment, apply_increment, finalization_offsets);
}

std::shared_ptr<LoopBegin> LoopEnd::get_loop_begin() {
    const auto& loop_begin = ov::as_type_ptr<LoopBegin>(get_input_source_output(get_input_size() - 1).get_node_shared_ptr());
    if (!loop_begin)
        throw std::invalid_argument("LoopEnd last input is not connected to LoopBegin");
    return  loop_begin;
}

const std::vector<int64_t>& LoopEnd::get_finalization_offsets() const {
    return finalization_offsets;
}

const std::vector<bool>& LoopEnd::get_apply_increment() const {
    return apply_increment;
}

void LoopEnd::set_finalization_offsets(std::vector<int64_t> offsets) {
    if (offsets.size() != loop_io_size)
        throw std::invalid_argument("LoopEnd set_finalization_offsets is called with inconsistent offsets.size()");
    finalization_offsets = std::move(offsets);
}

void LoopEnd::set_apply_increment(std::vector<bool> allow_increment) {
    if (allow_increment.size() != loop_io_size)
        throw std::invalid_argument("LoopEnd set_apply_increment is called with inconsistent apply_increment.size()");
    apply_increment = std::move(allow_increment);
}

void LoopEnd::set_work_amount(size_t new_work_amount) {
    work_amount = new_work_amount;
    // Update LoopBegin to maintain consistency between the Loops
    get_loop_begin()->work_amount = new_work_amount;
}

void LoopEnd::set_increment(size_t new_increment) {
    increment = new_increment;
    // Update LoopBegin to maintain consistency between the Loops
    get_loop_begin()->increment = new_increment;
}

void LoopEnd::set_evaluate_once(bool once) {
    evaluate_once = once;
    // Update LoopBegin to maintain consistency between the Loops
    get_loop_begin()->evaluate_once = once;
}

void LoopEnd::validate_and_infer_types() {
    const size_t num_inputs = get_input_size();
    const auto loop_begin = ov::as_type_ptr<LoopBegin>(input(get_input_size() - 1).get_source_output().get_node_shared_ptr());
    NODE_VALIDATION_CHECK(this, loop_begin != nullptr, "LoopEnd must have LoopBegin as the last argument");
    // Note: have to -2 because the LoopBegin->LoopEnd edge is counted twice
    loop_io_size = get_input_size() + loop_begin->get_output_size() - 2;
    NODE_VALIDATION_CHECK(this, apply_increment.empty() || apply_increment.size() == loop_io_size,
                          "apply_increments must be either empty or defined per every input & output of joined Loop. Expected size: ",
                          loop_io_size, " got ", apply_increment.size());
    NODE_VALIDATION_CHECK(this, finalization_offsets.empty() || finalization_offsets.size() == loop_io_size,
                          "finalization_offsets must be either empty or defined per every input & output of joined Loop. Expected size: ",
                          loop_io_size, " got ", finalization_offsets.size());
    if (apply_increment.empty())
        apply_increment.resize(loop_io_size, true);
    if (finalization_offsets.empty())
        finalization_offsets.resize(loop_io_size, 0);
    set_output_size(num_inputs - 1);
    const auto& ins = inputs();
    // All outputs are by-passed from inputs, except for the last one - it connects LoopBegin and LoopEnd
    for (int i = 0; i < num_inputs - 1; i++)
        get_output_descriptor(i).set_tensor_ptr(get_input_descriptor(i).get_output().get_tensor_ptr());
}

} // namespace op
} // namespace snippets
} // namespace ngraph