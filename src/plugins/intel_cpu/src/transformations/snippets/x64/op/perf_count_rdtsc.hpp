// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef SNIPPETS_DEBUG_CAPS

#pragma once

#include "openvino/op/op.hpp"
#include "snippets/op/perf_count.hpp"

using namespace ov::snippets::op;

namespace ov {
namespace intel_cpu {

/**
 * @interface PerfCountRdtscBegin
 * @brief Performance count start time via read rdtsc register
 * @ingroup snippets
 */
class PerfCountRdtscBegin : public PerfCountBeginBase {
public:
    OPENVINO_OP("PerfCountRdtscBegin", "SnippetsOpset", PerfCountBeginBase);
    PerfCountRdtscBegin();
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    uint64_t start_count = 0ul;
};

/**
 * @interface PerfCountRdtscEnd
 * @brief Performance count end time and duration
 * @ingroup snippets
 */
class PerfCountRdtscEnd : public PerfCountEndBase {
public:
    OPENVINO_OP("PerfCountRdtscEnd", "SnippetsOpset", PerfCountEndBase);
    PerfCountRdtscEnd(const Output<Node>& pc_begin);
    PerfCountRdtscEnd() = default;
    ~PerfCountRdtscEnd() {
        uint64_t avg = iteration == 0 ? 0 : accumulation / iteration;
        std::cout << "accumulation:" << accumulation << " iteration:" << iteration << " avg:" << avg << std::endl;
    }
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    std::shared_ptr<PerfCountRdtscBegin> get_pc_begin();
    // in each call, PerfCountRdtscBegin get start_count.
    // in each call, PerfCountRdtscEnd get end_count, then total_duration += end_count - start_count, and iteration++.
    // in destructor of PerfCountRdtscEnd, output the perf info
    // accumulation is cycle count
    uint64_t accumulation = 0ul;
    uint32_t iteration = 0u;
};

} // namespace intel_cpu
} // namespace ov
#endif // SNIPPETS_DEBUG_CAPS
