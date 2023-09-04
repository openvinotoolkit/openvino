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

PerfCountEnd::PerfCountEnd(PerfCountBegin& start) : Op(), perf_count_start(start), accumulation(0ul), iteration(0u) {}

std::shared_ptr<Node> PerfCountEnd::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<PerfCountEnd>(perf_count_start);
}

} // namespace op
} // namespace snippets
} // namespace ov
