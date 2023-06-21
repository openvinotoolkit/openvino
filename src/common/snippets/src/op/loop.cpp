// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/loop.hpp"
#include "snippets/generator.hpp"

namespace ov {
namespace snippets {
namespace op {

LoopBase::LoopBase(const std::vector<Output<Node>> &args) : Op(args) {
}

LoopBegin::LoopBegin() : LoopBase(), begin_address(nullptr) {
    validate_and_infer_types_except_LoopEnd();
}

std::shared_ptr<Node> LoopBegin::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<LoopBegin>();
}

void LoopBegin::validate_and_infer_types_except_LoopEnd() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 0, "LoopBegin doen't expect any inputs");
    set_output_type(0, element::f32, ov::PartialShape{ov::Shape{}});
}

void LoopBegin::validate_and_infer_types() {
    validate_and_infer_types_except_LoopEnd();
    OPENVINO_ASSERT(get_output_size() == 1, "LoopBegin must have only one output");
    const auto& last_output_inputs = get_output_target_inputs(0);
    OPENVINO_ASSERT(last_output_inputs.size() == 1, "LoopBegin must have exactly one input attached to the last output");
    const auto& loop_end = ov::as_type_ptr<LoopEnd>(last_output_inputs.begin()->get_node()->shared_from_this());
    OPENVINO_ASSERT(loop_end != nullptr, "LoopBegin must have LoopEnd connected to its last output");
}

std::shared_ptr<LoopEnd> LoopBegin::get_loop_end() const {
    const auto& last_output_inputs = get_output_target_inputs(0);
    if (last_output_inputs.size() != 1)
        throw std::invalid_argument("LoopBegin has more than one inputs attached to the last output");
    const auto& loop_end = ov::as_type_ptr<LoopEnd>(last_output_inputs.begin()->get_node()->shared_from_this());
    if (!loop_end)
        throw std::invalid_argument("LoopBegin last output is not connected to LoopEnd");
    return  loop_end;
}

bool LoopBegin::visit_attributes(AttributeVisitor &visitor) {
    return true;
}

LoopEnd::LoopEnd(const Output<Node>& loop_begin, size_t work_amount, size_t work_amount_increment,
                 std::vector<bool> apply_increments, std::vector<int64_t> finalization_offsets,
                 std::vector<int64_t> element_type_sizes, size_t input_num, size_t output_num)
        : LoopBase({loop_begin}),
        has_outer_loop(true),
        finalization_offsets(std::move(finalization_offsets)),
        element_type_sizes(std::move(element_type_sizes)),
        work_amount(work_amount),
        work_amount_increment(work_amount_increment),
        input_num(input_num),
        output_num(output_num),
        evaluate_once(false) {
        ptr_increments.resize(apply_increments.size());
        std::transform(apply_increments.begin(), apply_increments.end(), ptr_increments.begin(),
                       [](bool apply) {
                           return apply ? 1 : 0;
                       });
    constructor_validate_and_infer_types();
}

LoopEnd::LoopEnd(const Output<Node>& loop_begin, size_t work_amount, size_t work_amount_increment,
                 std::vector<int64_t> ptr_increments, std::vector<int64_t> finalization_offsets,
                 std::vector<int64_t> element_type_sizes, size_t input_num, size_t output_num)
        : LoopBase({loop_begin}),
        has_outer_loop(true),
        ptr_increments(std::move(ptr_increments)),
        finalization_offsets(std::move(finalization_offsets)),
        element_type_sizes(std::move(element_type_sizes)),
        work_amount(work_amount),
        work_amount_increment(work_amount_increment),
        input_num(input_num),
        output_num(output_num),
        evaluate_once(false) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> LoopEnd::clone_with_new_inputs(const OutputVector& inputs) const {
    check_new_args_count(this, inputs);
    return std::make_shared<LoopEnd>(inputs.at(0), work_amount, work_amount_increment, ptr_increments,
                                     finalization_offsets, element_type_sizes, input_num, output_num);
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

const std::vector<int64_t>& LoopEnd::get_ptr_increments()const {
    return ptr_increments;
}

const std::vector<int64_t>& LoopEnd::get_element_type_sizes() const {
    return element_type_sizes;
}

size_t LoopEnd::get_input_num() const {
    return input_num;
}

size_t LoopEnd::get_output_num() const {
    return output_num;
}

void LoopEnd::set_finalization_offsets(std::vector<int64_t> offsets) {
    OPENVINO_ASSERT(offsets.size() == input_num + output_num,
                    "LoopEnd set_finalization_offsets is called with inconsistent offsets.size()");
    finalization_offsets = std::move(offsets);
}

void LoopEnd::set_ptr_increments(std::vector<int64_t> new_ptr_increments) {
    OPENVINO_ASSERT(new_ptr_increments.size() == input_num + output_num,
                    "LoopEnd set_finalization_offsets is called with inconsistent offsets.size()");
    ptr_increments = std::move(new_ptr_increments);
}

void LoopEnd::update_ptr_increments(int64_t new_increment) {
    std::transform(ptr_increments.begin(), ptr_increments.end(), ptr_increments.begin(),
                   [new_increment](int64_t old_increment){
                        return old_increment != 0 ? new_increment : 0;
                   });
}

void LoopEnd::set_work_amount(size_t new_work_amount) {
    work_amount = new_work_amount;
}

void LoopEnd::set_increment(size_t new_increment) {
    work_amount_increment = new_increment;
}

void LoopEnd::set_evaluate_once(bool once) {
    evaluate_once = once;
}

void LoopEnd::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 1, "LoopEnd must have one input");
    const auto loop_begin = ov::as_type_ptr<LoopBegin>(get_input_node_shared_ptr(0));
    const auto io_size = input_num + output_num;
    NODE_VALIDATION_CHECK(this, loop_begin != nullptr, "LoopEnd must have LoopBegin as the last argument");
    NODE_VALIDATION_CHECK(this, ptr_increments.empty() || ptr_increments.size() == io_size,
                          "ptr_increments must be either empty or defined per every input & output of joined Loop. Expected size: ",
                          io_size, " got ", ptr_increments.size());
    NODE_VALIDATION_CHECK(this, finalization_offsets.empty() || finalization_offsets.size() == io_size,
                          "finalization_offsets must be either empty or defined per every input & output of joined Loop. Expected size: ",
                          io_size, " got ", finalization_offsets.size());
    if (ptr_increments.empty())
        ptr_increments.resize(io_size, 1);
    if (finalization_offsets.empty())
        finalization_offsets.resize(io_size, 0);
    set_output_type(0, element::f32, ov::PartialShape{ov::Shape{}});
}

bool LoopEnd::visit_attributes(AttributeVisitor &visitor) {
    visitor.on_attribute("work_amount", work_amount);
    visitor.on_attribute("increment", work_amount_increment);
    visitor.on_attribute("ptr_incr", ptr_increments);
    visitor.on_attribute("fin_offset", finalization_offsets);
    return true;
}

size_t LoopEnd::get_work_amount() const {
    return work_amount;
}

bool LoopEnd::get_evaluate_once() const {
    return evaluate_once;
}

size_t LoopEnd::get_increment() const {
    return work_amount_increment;
}

} // namespace op
} // namespace snippets
} // namespace ov