// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef SNIPPETS_DEBUG_CAPS

#    include "perf_count_rdtsc.hpp"

using namespace ov;
using namespace ov::intel_cpu;

/////////////////////////PerfCountRdtscBegin//////////////////////
PerfCountRdtscBegin::PerfCountRdtscBegin() : PerfCountBeginBase() {
    validate_and_infer_types_except_PerfCountEnd();
}

std::shared_ptr<Node> PerfCountRdtscBegin::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<PerfCountRdtscBegin>();
}

/////////////////////////PerfCountRdtscEnd//////////////////////
PerfCountRdtscEnd::PerfCountRdtscEnd(const Output<Node>& pc_begin) : ov::snippets::op::PerfCountEndBase({pc_begin}) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> PerfCountRdtscEnd::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<PerfCountRdtscEnd>(inputs.at(0));
}

std::shared_ptr<PerfCountRdtscBegin> PerfCountRdtscEnd::get_pc_begin() {
    const auto& pc_begin =
        ov::as_type_ptr<PerfCountRdtscBegin>(get_input_source_output(get_input_size() - 1).get_node_shared_ptr());
    OPENVINO_ASSERT(pc_begin != nullptr, "PerfCountRdtscEnd last input is not connected to PerfCountRdtscBegin");
    return pc_begin;
}
#endif  // SNIPPETS_DEBUG_CAPS
