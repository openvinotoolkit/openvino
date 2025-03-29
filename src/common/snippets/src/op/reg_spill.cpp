// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/reg_spill.hpp"

#include "snippets/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace op {

RegSpillBase::RegSpillBase(const std::vector<Output<Node>> &args) : Op(args) {}

bool RegSpillBase::visit_attributes(AttributeVisitor &visitor) {
    std::stringstream ss;
    const auto& regs_to_spill = get_regs_to_spill();
    for (auto reg_it = regs_to_spill.begin(); reg_it != regs_to_spill.end(); reg_it++) {
        ss << *reg_it;
        if (std::next(reg_it) != regs_to_spill.end())
            ss << ", ";
    }
    std::string spilled = ss.str();
    visitor.on_attribute("regs_to_spill", spilled);
    return true;
}

RegSpillBegin::RegSpillBegin(std::set<Reg> regs_to_spill) : m_regs_to_spill(std::move(regs_to_spill)) {
    validate_and_infer_types_except_RegSpillEnd();
}

void RegSpillBegin::validate_and_infer_types_except_RegSpillEnd() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 0, "RegSpillBegin doesn't expect any inputs");
    set_output_type(0, element::f32, ov::PartialShape{ov::Shape{}});
}

void RegSpillBegin::validate_and_infer_types() {
    validate_and_infer_types_except_RegSpillEnd();
    OPENVINO_ASSERT(get_output_size() == 1, "RegSpillBegin must have only one output");
    const auto& last_output_inputs = get_output_target_inputs(0);
    OPENVINO_ASSERT(last_output_inputs.size() == 1, "RegSpillBegin must have exactly one input attached to the last output");
    OPENVINO_ASSERT(ov::is_type<RegSpillEnd>(last_output_inputs.begin()->get_node()),
                    "RegSpillBegin must have RegSpillEnd connected to its last output");
}

std::shared_ptr<Node> RegSpillBegin::clone_with_new_inputs(const OutputVector& inputs) const {
    OPENVINO_ASSERT(inputs.empty(), "RegSpillBegin should not contain inputs");
    return std::make_shared<RegSpillBegin>(m_regs_to_spill);
}

std::shared_ptr<RegSpillEnd> RegSpillBegin::get_reg_spill_end() const {
    const auto& last_output_inputs = get_output_target_inputs(0);
    OPENVINO_ASSERT(last_output_inputs.size() == 1, "RegSpillBegin has more than one inputs attached to the last output");
    const auto& loop_end = ov::as_type_ptr<RegSpillEnd>(last_output_inputs.begin()->get_node()->shared_from_this());
    OPENVINO_ASSERT(loop_end != nullptr, "RegSpillBegin must have RegSpillEnd connected to its last output");
    return loop_end;
}

RegSpillBegin::ShapeInfer::ShapeInfer(const std::shared_ptr<ov::Node>& n) {
    auto reg_spill_begin = ov::as_type_ptr<RegSpillBegin>(n);
    OPENVINO_ASSERT(reg_spill_begin, "Invalid node passed to RegSpillBegin::ShapeInfer");
    num_out_shapes = reg_spill_begin->get_regs_to_spill().size();
}

RegSpillBegin::ShapeInfer::Result RegSpillBegin::ShapeInfer::infer(const std::vector<VectorDimsRef>& input_shapes) {
    return {std::vector<VectorDims>(num_out_shapes, VectorDims{1}), ShapeInferStatus::success};
}

RegSpillEnd::RegSpillEnd(const Output<Node>& reg_spill_begin) : RegSpillBase({reg_spill_begin}) {
    constructor_validate_and_infer_types();
}

void RegSpillEnd::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 1 && ov::is_type<RegSpillBegin>(get_input_node_shared_ptr(0)),
                         "RegSpillEnd must have one input of RegSPillBegin type");
    set_output_type(0, element::f32, ov::PartialShape{});
}

std::shared_ptr<Node> RegSpillEnd::clone_with_new_inputs(const OutputVector& inputs) const {
    check_new_args_count(this, inputs);
    return std::make_shared<RegSpillEnd>(inputs.at(0));
}


} // namespace op
} // namespace snippets
} // namespace ov
