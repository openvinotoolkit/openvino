// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

void core_configuration(ov::test::SubgraphBaseTest* test) {
    auto& config = test->configuration;
    auto exec_mode = config.find(hint::execution_mode.name());
    auto inf_prc = config.find(hint::inference_precision.name());

    // Force fp32 inference precision if it is not configured specially
    if (inf_prc == config.end() &&
        (exec_mode == config.end() || exec_mode->second != hint::ExecutionMode::ACCURACY)) {
        config.insert({hint::inference_precision.name(), element::f32.to_string()});
    }

    // todo: issue: 123320
    if (!((inf_prc != config.end() && inf_prc->second == element::undefined)
        || (inf_prc == config.end() && exec_mode != config.end() && exec_mode->second == hint::ExecutionMode::ACCURACY))) {
        test->convert_precisions.insert({ov::element::bf16, ov::element::f32});
        test->convert_precisions.insert({ov::element::f16, ov::element::f32});
    }

    // Enable CPU pinning in CPU funtional tests to save validation time of Intel CPU plugin func tests (parallel)
    // on Windows
    config.insert({ov::hint::enable_cpu_pinning.name(), true});
}

} // namespace test
} // namespace ov
