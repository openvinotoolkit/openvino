// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "execution_graph_tests/disable_lowering_precision.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/runtime/system_conf.hpp"

using namespace ExecutionGraphTests;

namespace {

const std::vector<ExecGraphDisableLoweringPrecisionSpecificParams> disableLoweringPrecisionTestParams = {
    {true,  "CPU", ov::element::bf16},
    {false, "CPU", ov::element::bf16},
};

INSTANTIATE_TEST_SUITE_P(smoke_ExecGraph, ExecGraphDisableLoweringPrecision,
                                // Only run tests on CPU with avx512_core ISA
                                ::testing::ValuesIn(ov::with_cpu_x86_avx512_core() ?
                                                        disableLoweringPrecisionTestParams :
                                                        std::vector<ExecGraphDisableLoweringPrecisionSpecificParams>{}),
                        ExecGraphDisableLoweringPrecision::getTestCaseName);

}  // namespace
