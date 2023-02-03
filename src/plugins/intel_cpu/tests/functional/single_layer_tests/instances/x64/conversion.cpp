// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/conversion.hpp"
#include "shared_test_classes/single_layer/conversion.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>

using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
namespace Conversion {
namespace {

std::vector<CPUSpecificParams> memForm4D_dynamic = {
    CPUSpecificParams({nChw8c}, {nChw8c}, {}, "ref"),
    CPUSpecificParams({nChw16c}, {nChw16c}, {}, "ref")
};

ov::AnyMap empty_config = {};
ov::AnyMap config_i64 = {{InferenceEngine::PluginConfigInternalParams::KEY_CPU_NATIVE_I64, InferenceEngine::PluginConfigParams::YES}};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_blocked_Dynamic, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_dynamic()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::Values(empty_config),
                                ::testing::ValuesIn(memForm4D_dynamic)),
                        ConvertCPULayerTest::getTestCaseName);

std::vector<InputShape> inShapes_4D_blocked = {
    {{1, 16, 5, 5}, {{1, 16, 5, 5}}},
};

std::vector<CPUSpecificParams> memForm4D_static_blocked = {
    CPUSpecificParams({nChw16c}, {nChw16c}, {}, {})
};

const std::vector<ElementType> precisions_floating_point = {
        ElementType::f32,
        ElementType::bf16
};

std::vector<CPUSpecificParams> memForm4D_static_common = {
    CPUSpecificParams({nchw}, {nchw}, {}, {}),
    CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_Blocked, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_blocked),
                                ::testing::ValuesIn(precisions()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::Values(empty_config),
                                ::testing::ValuesIn(filterCPUSpecificParams(memForm4D_static_blocked))),
                        ConvertCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_BOOL_Static, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_static()),
                                ::testing::ValuesIn(precisions_floating_point),
                                ::testing::Values(ElementType::boolean),
                                ::testing::Values(empty_config),
                                ::testing::Values(CPUSpecificParams({nchw}, {nchw}, {}, {}))),
                        ConvertCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_BOOL_Dynamic, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_dynamic()),
                                ::testing::ValuesIn(precisions_floating_point),
                                ::testing::Values(ElementType::boolean),
                                ::testing::Values(empty_config),
                                ::testing::Values(CPUSpecificParams({nchw}, {nchw}, {}, "ref"))),
                        ConvertCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_FromI64_Dynamic, ConvertCPULayerTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inShapes_4D_dynamic()),
                                 ::testing::Values(ElementType::i64),
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(config_i64),
                                 ::testing::ValuesIn(memForm4D_dynamic)),
                         ConvertCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_ToI64_Dynamic, ConvertCPULayerTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inShapes_4D_dynamic()),
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(ElementType::i64),
                                 ::testing::Values(config_i64),
                                 ::testing::ValuesIn(memForm4D_dynamic)),
                         ConvertCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_FromI64, ConvertCPULayerTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inShapes_4D_static()),
                                 ::testing::Values(ElementType::i64),
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(config_i64),
                                 ::testing::ValuesIn(memForm4D_static_common)),
                         ConvertCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_ToI64, ConvertCPULayerTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inShapes_4D_static()),
                                 ::testing::ValuesIn(precisions()),
                                 ::testing::Values(ElementType::i64),
                                 ::testing::Values(config_i64),
                                 ::testing::ValuesIn(memForm4D_static_common)),
                         ConvertCPULayerTest::getTestCaseName);

}  // namespace
}  // namespace Conversion
}  // namespace CPULayerTestsDefinitions