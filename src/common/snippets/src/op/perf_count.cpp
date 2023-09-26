// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/perf_count.hpp"

namespace ov {
namespace snippets {
namespace op {

/////////////////PerfCountBeginBase/////////////////
PerfCountBeginBase::PerfCountBeginBase(const std::vector<Output<Node>>& args) : Op() {}

void PerfCountBeginBase::validate_and_infer_types() {
    validate_and_infer_types_except_PerfCountEnd();
    OPENVINO_ASSERT(get_output_size() == 1, "PerfCountBegin must have only one output");
    const auto& last_output_inputs = get_output_target_inputs(0);
    OPENVINO_ASSERT(last_output_inputs.size() == 1, "PerfCountBegin must have exactly one input attached to the last output");
    const auto& pc_end = ov::as_type_ptr<PerfCountEnd>(last_output_inputs.begin()->get_node()->shared_from_this());
    OPENVINO_ASSERT(pc_end != nullptr, "PerfCountBegin must have PerfCountEnd connected to its last output");
}

bool PerfCountBeginBase::visit_attributes(AttributeVisitor &visitor) {
    return true;
}

void PerfCountBeginBase::validate_and_infer_types_except_PerfCountEnd() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 0, "PerfCountBegin dosen't expect any inputs");
    set_output_type(0, element::f32, ov::PartialShape{ov::Shape{}});
}

//////////////////PerfCountEndBase/////////////////
PerfCountEndBase::PerfCountEndBase(const std::vector<Output<Node>> &args) : Op(args) {}

void PerfCountEndBase::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 1, "PerfCountEndBase must have one input");
    const auto pc_begin = ov::as_type_ptr<PerfCountBegin>(get_input_node_shared_ptr(0));
    NODE_VALIDATION_CHECK(this, pc_begin != nullptr, "PerfCountEndBase must have PerfCountBegin as the last argument");
    set_output_type(0, element::f32, ov::PartialShape{ov::Shape{}});
}

bool PerfCountEndBase::visit_attributes(AttributeVisitor &visitor) {
    return true;
}

/////////////////PerfCountBegin/////////////////
PerfCountBegin::PerfCountBegin() : PerfCountBeginBase() {
    validate_and_infer_types_except_PerfCountEnd();
}

std::shared_ptr<Node> PerfCountBegin::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<PerfCountBegin>();
}

std::chrono::high_resolution_clock::time_point& PerfCountBegin::get_start_time() {
    return start_time_stamp;
}

void PerfCountBegin::set_start_time() {
    start_time_stamp = std::chrono::high_resolution_clock::now();
}

//////////////////PerfCountEnd///////////////
PerfCountEnd::PerfCountEnd(const Output<Node>& pc_begin) : PerfCountEndBase({pc_begin}), accumulation(0ul), iteration(0u) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> PerfCountEnd::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<PerfCountEnd>(inputs.at(0));
}

void PerfCountEnd::set_accumulated_time() {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto& start_time = get_pc_begin()->get_start_time();
    accumulation += std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - start_time).count();
    iteration++;
}

std::shared_ptr<PerfCountBegin> PerfCountEnd::get_pc_begin() {
    const auto& pc_begin = ov::as_type_ptr<PerfCountBegin>(get_input_source_output(get_input_size() - 1).get_node_shared_ptr());
    if (!pc_begin)
        throw std::invalid_argument("PerfCountEnd last input is not connected to PerfCountBegin");
    return  pc_begin;
}

} // namespace op
} // namespace snippets
} // namespace ov
