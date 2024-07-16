// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/conversion.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Conversion {
namespace {

std::vector<CPUSpecificParams> memForm4D_dynamic = {
    CPUSpecificParams({nChw8c}, {nChw8c}, {}, "ref"),
    CPUSpecificParams({nChw16c}, {nChw16c}, {}, "ref")
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_blocked_Dynamic, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_dynamic()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::ValuesIn(memForm4D_dynamic)),
                        ConvertCPULayerTest::getTestCaseName);

std::vector<InputShape> inShapes_4D_blocked = {
    {{1, 16, 5, 5}, {{1, 16, 5, 5}}},
};

std::vector<CPUSpecificParams> memForm4D_static_blocked = {
    CPUSpecificParams({nChw16c}, {nChw16c}, {}, {})
};

const std::vector<ov::element::Type> precisions_floating_point = {
        ov::element::f32,
        ov::element::bf16
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_Blocked, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_blocked),
                                ::testing::ValuesIn(precisions()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::ValuesIn(filterCPUSpecificParams(memForm4D_static_blocked))),
                        ConvertCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_BOOL_Static, ConvertToBooleanCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_static()),
                                ::testing::ValuesIn(precisions_floating_point),
                                ::testing::Values(ov::element::boolean),
                                ::testing::Values(CPUSpecificParams({nchw}, {nchw}, {}, {}))),
                        ConvertToBooleanCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_BOOL_Dynamic, ConvertToBooleanCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_dynamic()),
                                ::testing::ValuesIn(precisions_floating_point),
                                ::testing::Values(ov::element::boolean),
                                ::testing::Values(CPUSpecificParams({nchw}, {nchw}, {}, {}))),
                        ConvertToBooleanCPULayerTest::getTestCaseName);

}  // namespace
}  // namespace Conversion
}  // namespace test
}  // namespace ov