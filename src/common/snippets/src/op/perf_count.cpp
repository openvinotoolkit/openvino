// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/loop.hpp"
#include "snippets/generator.hpp"

namespace ov {
namespace snippets {
namespace op {

PerfCountBegin::PerfCountBegin() : Op() {}

PerfCountEnd::PerfCountEnd(PerfCountBegin& start) : Op(), perf_count_start(start), accumulation(0ul), iteration(0ul) {}

} // namespace op
} // namespace snippets
} // namespace ov