// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/loop.hpp"
#include "snippets/generator.hpp"

namespace ov {
namespace snippets {
namespace op {

PerfCountBegin::PerfCountBegin() : Op() {}

std::shared_ptr<Node> PerfCountBegin::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<PerfCountBegin>();
}

PerfCountEnd::PerfCountEnd(const Output<Node>& start) : Op({start}), accumulation(0ul), iteration(0ul) {}

std::shared_ptr<PerfCountBegin> PerfCountEnd::get_perf_count_start() {
    const auto& perf_count_begin = ov::as_type_ptr<PerfCountBegin>(get_input_source_output(0).get_node_shared_ptr());
    if (!perf_count_begin)
        throw std::invalid_argument("PerfCountEnd is not connected to PerfCountBegin");
    return  perf_count_begin;
}

std::shared_ptr<Node> PerfCountEnd::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<PerfCountEnd>(inputs.at(0));
}

} // namespace op
} // namespace snippets
} // namespace ov
