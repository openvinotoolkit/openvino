// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/extremum.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace extremum {

const auto basicCasesSnippets = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(inputShape())),
    ::testing::ValuesIn(extremumTypes()),
    ::testing::ValuesIn(netPrecisions()),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams4D())),
    ::testing::Values(true)
);

INSTANTIATE_TEST_SUITE_P(smoke_Extremum_Snippets_CPU, ExtremumLayerCPUTest, basicCasesSnippets, ExtremumLayerCPUTest::getTestCaseName);

}  // namespace extremum
}  // namespace test
}  // namespace ov
