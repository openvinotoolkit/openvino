// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace snippets {
namespace op {

/**
 * @interface PerfCountBegin
 * @brief Performance count start time
 * @ingroup snippets
 */
class PerfCountBegin : public ov::op::Op {
public:
    OPENVINO_OP("PerfCountBegin", "SnippetsOpset");
    PerfCountBegin();

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    // used for chrono call
    std::chrono::high_resolution_clock::time_point start_time_stamp = {};

    // used for rdtsc
    uint64_t start_count = 0ul;
};

/**
 * @interface PerfCountEnd
 * @brief Performance count end time and duration
 * @ingroup snippets
 */
class PerfCountBegin;
class PerfCountEnd : public ov::op::Op {
public:
    OPENVINO_OP("PerfCountEnd", "SnippetsOpset");
    PerfCountEnd(PerfCountBegin& start);
    ~PerfCountEnd() {
        uint64_t avg = iteration == 0 ? 0 : accumulation / iteration;
        std::cout << "accumulation:" << accumulation << " iteration:" << iteration << " avg:" << avg << std::endl;
    }
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    // in each call, PerfCountBegin get start_time_stamp.
    // in each call, PerfCountEnd get end_time_stamp, then total_duration += end_time_stamp - start_time_stamp, and iteration++.
    // in destructor of PerfCountEnd, output the perf info
    PerfCountBegin& perf_count_start;
    // accumulation is time in nanosecond for chrono call, cycle count for rdtsc
    uint64_t accumulation = 0ul;
    uint32_t iteration = 0u;
};

} // namespace op
} // namespace snippets
} // namespace ov
