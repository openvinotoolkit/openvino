// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/pooling.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Pooling {

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolV14_CPU_4D, MaxPoolingV14LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsMaxV144D()),
                                 ::testing::ValuesIn(inputShapes4D()),
                                 ::testing::ValuesIn((inpOutPrecision())),
                                 ::testing::ValuesIn(filterCPUInfo({CPUSpecificParams{{}, {}, {"ref_any"}, "ref_any"}})),
                                 ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MaxPoolingV14LayerCPUTest::getTestCaseName);

}  // namespace Pooling
}  // namespace test
}  // namespace ov