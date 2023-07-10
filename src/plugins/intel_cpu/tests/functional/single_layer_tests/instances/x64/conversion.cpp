// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/conversion.hpp"
#include "shared_test_classes/single_layer/conversion.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
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

const std::vector<Precision> precisions_floating_point = {
        Precision::FP32,
        Precision::BF16
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_Blocked, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_blocked),
                                ::testing::ValuesIn(precisions()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::ValuesIn(filterCPUSpecificParams(memForm4D_static_blocked))),
                        ConvertCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_BOOL_Static, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_static()),
                                ::testing::ValuesIn(precisions_floating_point),
                                ::testing::Values(Precision::BOOL),
                                ::testing::Values(CPUSpecificParams({nchw}, {nchw}, {}, {}))),
                        ConvertCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_BOOL_Dynamic, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_dynamic()),
                                ::testing::ValuesIn(precisions_floating_point),
                                ::testing::Values(Precision::BOOL),
                                ::testing::Values(CPUSpecificParams({nchw}, {nchw}, {}, "ref"))),
                        ConvertCPULayerTest::getTestCaseName);

}  // namespace
}  // namespace Conversion
}  // namespace CPULayerTestsDefinitions