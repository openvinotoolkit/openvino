// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef SNIPPETS_DEBUG_CAPS

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
    const auto& pc_end = ov::as_type_ptr<PerfCountEndBase>(last_output_inputs.begin()->get_node()->shared_from_this());
    OPENVINO_ASSERT(pc_end != nullptr, "PerfCountBegin must have PerfCountEnd connected to its last output");
}

bool PerfCountBeginBase::visit_attributes(AttributeVisitor &visitor) {
    return true;
}

void PerfCountBeginBase::validate_and_infer_types_except_PerfCountEnd() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 0, "PerfCountBegin doesn't expect any inputs");
    set_output_type(0, element::f32, {});
}

//////////////////PerfCountEndBase/////////////////
PerfCountEndBase::PerfCountEndBase(const std::vector<Output<Node>> &args) : Op(args) {}

void PerfCountEndBase::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 1, "PerfCountEndBase must have one input");
    const auto& pc_begin = ov::as_type_ptr<PerfCountBeginBase>(get_input_node_shared_ptr(0));
    NODE_VALIDATION_CHECK(this, pc_begin != nullptr, "PerfCountEndBase must have PerfCountBeginBase as the last argument");
    set_output_type(0, element::f32, {});
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
    return start_time_stamp.local();
}

void PerfCountBegin::set_start_time() {
    start_time_stamp.local() = std::chrono::high_resolution_clock::now();
}

//////////////////PerfCountEnd///////////////
PerfCountEnd::PerfCountEnd(const Output<Node>& pc_begin) : PerfCountEndBase({pc_begin}), accumulation(0ul), iteration(0u) {
    constructor_validate_and_infer_types();
    init_pc_begin();
}

std::shared_ptr<Node> PerfCountEnd::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<PerfCountEnd>(inputs.at(0));
}

void PerfCountEnd::set_accumulated_time() {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto& start_time = m_pc_begin->get_start_time();
    accumulation.local() += std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - start_time).count();
    iteration.local()++;
}

void PerfCountEnd::init_pc_begin() {
    m_pc_begin = ov::as_type_ptr<PerfCountBegin>(get_input_source_output(get_input_size() - 1).get_node_shared_ptr());
    NODE_VALIDATION_CHECK(this, m_pc_begin != nullptr, "PerfCountEnd last input is not connected to PerfCountBegin");
}

void PerfCountEnd::output_perf_count() {
    OPENVINO_ASSERT(accumulation.size() == iteration.size(), "accumulation size should be the same as iteration size in perf_count_end node.");
    auto iterator_iter = iteration.begin();
    auto iterator_acc = accumulation.begin();
    int t_num = 0;
    uint64_t avg_max = 0;
    std::cout << "Perf count data in perfCountEnd node with name " << get_friendly_name() << " is:"<< std::endl;
    for (; iterator_iter != iteration.end(); ++iterator_iter, ++iterator_acc) {
        const auto iter = *iterator_iter;
        const auto acc = *iterator_acc;
        uint64_t avg = iter == 0 ? 0 : acc / iter;
        if (avg > avg_max)
            avg_max = avg;
        std::cout << "accumulated time:" << acc << "ns, iteration:" << iter << " avg time:" << avg << "ns"<< " on thread:" << t_num << std::endl;
        t_num++;
    }

    // max time of all threads: combine for reduce max
    auto BinaryFunc = [](const uint64_t& a, const uint64_t& b) {
        return a >= b ? a : b;
    };
    // max accumulation
    uint64_t acc_max = accumulation.combine(BinaryFunc);
    std::cout << "max accumulated time:" << acc_max << "ns" << std::endl;
    // max avg
    std::cout << "max avg time:" << avg_max << "ns" << std::endl;
}

} // namespace op
} // namespace snippets
} // namespace ov
#endif  // SNIPPETS_DEBUG_CAPS
