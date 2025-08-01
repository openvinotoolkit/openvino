// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cstdint>
#include <ios>
#include <iostream>
#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#ifdef SNIPPETS_DEBUG_CAPS

#    pragma once

#    include <iomanip>

#    include "openvino/op/op.hpp"
#    include "snippets/op/perf_count.hpp"

namespace ov::intel_cpu {

using namespace ov::snippets::op;

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

    uint64_t start_count = 0UL;
};

/**
 * @interface PerfCountRdtscEnd
 * @brief Performance count end time and duration
 * @ingroup snippets
 */
class PerfCountRdtscEnd : public PerfCountEndBase {
public:
    OPENVINO_OP("PerfCountRdtscEnd", "SnippetsOpset", PerfCountEndBase);
    explicit PerfCountRdtscEnd(const Output<Node>& pc_begin);
    PerfCountRdtscEnd() = default;
    ~PerfCountRdtscEnd() override {
        double avg = 0;
        if (iteration != 0) {
            // Note: theoretically accumulation could be larger than 2^53, however
            // iteration is unlikely to exceed this threshold. So here we derive an integral part first
            // and cast only the remainder to double
            const uint64_t integral = accumulation / iteration;
            avg = integral + static_cast<double>(accumulation - integral * iteration) / iteration;
        }
        std::cerr << "name : " << get_friendly_name() << " : acc : " << accumulation << " : num_hit : " << iteration
                  << std::fixed << std::setprecision(4) << " : avg : " << avg << '\n';
    }
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    std::shared_ptr<PerfCountRdtscBegin> get_pc_begin();
    // in each call, PerfCountRdtscBegin get start_count.
    // in each call, PerfCountRdtscEnd get end_count, then total_duration += end_count - start_count, and iteration++.
    // in destructor of PerfCountRdtscEnd, output the perf info
    // accumulation is cycle count
    uint64_t accumulation = 0UL;
    uint64_t iteration = 0UL;
};

}  // namespace ov::intel_cpu

#endif  // SNIPPETS_DEBUG_CAPS
