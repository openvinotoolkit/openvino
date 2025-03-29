// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef SNIPPETS_DEBUG_CAPS

#include <fstream>

#include "snippets/op/perf_count.hpp"

namespace ov {
namespace snippets {

//////////////////utils///////////////

namespace utils {

//////////////////utils::Dumper///////////////

void Dumper::init(const std::string &params) {
    m_params = params;
}

//////////////////utils::ConsoleDumper///////////////

ConsoleDumper::~ConsoleDumper() {
    OPENVINO_ASSERT(m_accumulation.size() == m_iteration.size(),
                    "accumulation size should be the same as iteration size in perf_count_end node.");
    auto iterator_iter = m_iteration.begin();
    auto iterator_acc = m_accumulation.begin();
    uint64_t avg_max = 0;
    for (; iterator_iter != m_iteration.end(); ++iterator_iter, ++iterator_acc) {
        const auto iter = *iterator_iter;
        const auto acc = *iterator_acc;
        uint64_t avg = iter == 0 ? 0 : acc / iter;
        if (avg > avg_max)
            avg_max = avg;
    }

    // max time of all threads: combine for reduce max
    auto BinaryFunc = [](const uint64_t& a, const uint64_t& b) {
        return a >= b ? a : b;
    };

    // max accumulation
    uint64_t acc_max = m_accumulation.combine(BinaryFunc);
    std::cout << "max accumulated time:" << acc_max << "ns" << std::endl;
    // max avg
    std::cout << "max avg time:" << avg_max << "ns" << std::endl;
}

void ConsoleDumper::update(const op::PerfCountEnd* node) {
    auto accumulation = node->get_accumulation();
    auto iteration = node->get_iteration();
    OPENVINO_ASSERT(accumulation.size() == iteration.size(),
                    "accumulation size should be the same as iteration size in perf_count_end node.");
    auto iterator_iter = iteration.begin();
    auto iterator_acc = accumulation.begin();
    for (; iterator_iter != iteration.end(); ++iterator_iter, ++iterator_acc) {
        m_accumulation.local() += *iterator_acc;
        m_iteration.local() += *iterator_iter;
    }
}

//////////////////utils::CSVDumper///////////////

CSVDumper::CSVDumper(const std::string csv_path) : csv_path(std::move(csv_path)) {}

CSVDumper::~CSVDumper() {
    if (m_debug_params_map.empty() || csv_path.empty()) {
        return;
    }
    std::ofstream csv_file(csv_path, std::ios_base::app);
    OPENVINO_ASSERT(csv_file.is_open(), "Failed to open csv file for brgemm debug parameters.");
    if (csv_file.tellp() == 0) {
        csv_file << "name,subgraph_name,in_type,out_type,in_shapes,out_shapes,in_layouts,out_layouts,M,N,K,m_block,n_"
                    "block,k_block,acc_max_time,"
                    "avg_max_time\n";
    }
    for (const auto& [_, params] : m_debug_params_map) {
        csv_file << params << '\n';
    }
    csv_file.close();
}

void CSVDumper::update(const op::PerfCountEnd* node) {
    auto accumulation = node->get_accumulation();
    auto iteration = node->get_iteration();
    OPENVINO_ASSERT(accumulation.size() == iteration.size(), "accumulation size should be the same as iteration size in perf_count_end node.");
    auto iterator_iter = iteration.begin();
    auto iterator_acc = accumulation.begin();
    uint64_t avg_max = 0;
    for (; iterator_iter != iteration.end(); ++iterator_iter, ++iterator_acc) {
        const auto iter = *iterator_iter;
        const auto acc = *iterator_acc;
        uint64_t avg = iter == 0 ? 0 : acc / iter;
        if (avg > avg_max)
            avg_max = avg;
    }

    // max time of all threads: combine for reduce max
    auto BinaryFunc = [](const uint64_t& a, const uint64_t& b) {
        return a >= b ? a : b;
    };

    // max accumulation
    uint64_t acc_max = accumulation.combine(BinaryFunc);

    m_debug_params_map[node->get_friendly_name()] = m_params + std::to_string(acc_max) + ',' + std::to_string(avg_max);
}

}  // namespace utils

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

PerfCountEnd::PerfCountEnd() : PerfCountEndBase() {}

PerfCountEnd::PerfCountEnd(const Output<Node>& pc_begin,
                           std::vector<std::shared_ptr<utils::Dumper>> dumpers,
                           const std::string& params)
    : PerfCountEndBase({pc_begin}),
      accumulation(0ul),
      iteration(0u),
      dumpers(std::move(dumpers)) {
    constructor_validate_and_infer_types();
    init_pc_begin();
    for (const auto& dumper : dumpers) {
        dumper->init(params);
    }
}

PerfCountEnd::~PerfCountEnd() {
    for (const auto& dumper : dumpers) {
        dumper->update(this);
    }
}

std::shared_ptr<Node> PerfCountEnd::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<PerfCountEnd>(inputs.at(0), dumpers);
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

} // namespace op
} // namespace snippets
} // namespace ov
#endif  // SNIPPETS_DEBUG_CAPS
