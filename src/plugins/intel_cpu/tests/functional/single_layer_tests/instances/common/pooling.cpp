// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/pooling.hpp"
#include "shared_test_classes/single_layer/pooling.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
namespace Pooling {

static CPUSpecificParams expectedCpuConfig() {
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    return CPUSpecificParams{{}, {}, {"acl"}, "acl"};
#else
    return CPUSpecificParams{{}, {}, {"ref_any"}, "ref_any"};
#endif
}
const std::vector<CPUSpecificParams> vecCpuConfigs = {expectedCpuConfig()};

const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg3D_RefOnly = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2}, {2}, {2}, {2},
                            expectedAvgRoundingType(), ngraph::op::PadType::EXPLICIT, false },
};

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool_CPU_3D, PoolingLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsMax3D()),
                                 ::testing::ValuesIn(inputShapes3D()),
                                 ::testing::ValuesIn((inpOutPrecision())),
                                 ::testing::Values(false),
                                 ::testing::ValuesIn(vecCpuConfigs),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_3D, PoolingLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsAvg3D()),
                                 ::testing::ValuesIn(inputShapes3D()),
                                 ::testing::ValuesIn((inpOutPrecision())),
                                 ::testing::Values(false),
                                 ::testing::ValuesIn(vecCpuConfigs),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_3D_NotOptimized, PoolingLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsAvg3D_RefOnly),
                                 ::testing::ValuesIn(inputShapes3D()),
                                 ::testing::ValuesIn((inpOutPrecision())),
                                 ::testing::Values(false),
                                 ::testing::Values(expectedCpuConfig()),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         PoolingLayerCPUTest::getTestCaseName);

const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg4D_RefOnly = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {2, 2}, {2, 2},
                            expectedAvgRoundingType(), ngraph::op::PadType::EXPLICIT, false },
};

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool_CPU_4D, PoolingLayerCPUTest,
                            ::testing::Combine(
                            ::testing::ValuesIn(paramsMax4D()),
                            ::testing::ValuesIn(inputShapes4D()),
                            ::testing::ValuesIn((inpOutPrecision())),
                            ::testing::Values(false),
                            ::testing::ValuesIn(vecCpuConfigs),
                            ::testing::Values(emptyFusingSpec),
                            ::testing::Values(cpuEmptyPluginConfig)),
                        PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolV8_CPU_4D, MaxPoolingV8LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsMaxV84D()),
                                 ::testing::ValuesIn(inputShapes4D()),
                                 ::testing::ValuesIn((inpOutPrecision())),
                                 ::testing::ValuesIn(vecCpuConfigs),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         MaxPoolingV8LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_4D, PoolingLayerCPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(paramsAvg4D()),
                            ::testing::ValuesIn(inputShapes4D()),
                            ::testing::ValuesIn((inpOutPrecision())),
                            ::testing::Values(false),
                            ::testing::ValuesIn(vecCpuConfigs),
                            ::testing::Values(emptyFusingSpec),
                            ::testing::Values(cpuEmptyPluginConfig)),
                        PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_4D_NotOptimized, PoolingLayerCPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(paramsAvg4D_RefOnly),
                            ::testing::ValuesIn(inputShapes4D()),
                            ::testing::ValuesIn((inpOutPrecision())),
                            ::testing::Values(false),
                            ::testing::Values(expectedCpuConfig()),
                            ::testing::Values(emptyFusingSpec),
                            ::testing::Values(cpuEmptyPluginConfig)),
                        PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_Large, PoolingLayerCPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(paramsAvg4D_Large()),
                            ::testing::ValuesIn(inputShapes4D_Large()),
                            ::testing::ValuesIn((inpOutPrecision())),
                            ::testing::Values(false),
                            ::testing::ValuesIn(vecCpuConfigs),
                            ::testing::Values(emptyFusingSpec),
                            ::testing::Values(cpuEmptyPluginConfig)),
                        PoolingLayerCPUTest::getTestCaseName);

const std::vector<LayerTestsDefinitions::maxPoolV8SpecificParams> paramsMaxV85D_ref = {
        LayerTestsDefinitions::maxPoolV8SpecificParams{ {2, 2, 2}, {1, 1, 1}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0},
                                                        ngraph::element::Type_t::i32, 0,
                                                        ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER },
        LayerTestsDefinitions::maxPoolV8SpecificParams{ {2, 2, 2}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1},
                                                        ngraph::element::Type_t::i32, 0,
                                                        ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT },
        LayerTestsDefinitions::maxPoolV8SpecificParams{ {2, 3, 4}, {2, 2, 2}, {2, 1, 1}, {1, 1, 1}, {1, 2, 2},
                                                        ngraph::element::Type_t::i32, 0,
                                                        ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT },
};

const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg5D_RefOnly = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2},
                            expectedAvgRoundingType(), ngraph::op::PadType::EXPLICIT, false },
};

//FIXME: 5D cases are temporarly disabled on ARM because ACL support check in Pooling::getSupportedDescriptors() can't check layout
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
INSTANTIATE_TEST_SUITE_P(smoke_MaxPool_CPU_5D, PoolingLayerCPUTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(paramsMax5D()),
                             ::testing::ValuesIn(inputShapes5D()),
                             ::testing::ValuesIn((inpOutPrecision())),
                             ::testing::Values(false),
                             ::testing::ValuesIn(vecCpuConfigs),
                             ::testing::Values(emptyFusingSpec),
                             ::testing::Values(cpuEmptyPluginConfig)),
                         PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolV8_CPU_5D, MaxPoolingV8LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsMaxV85D()),
                                 ::testing::ValuesIn(inputShapes5D()),
                                 ::testing::ValuesIn((inpOutPrecision())),
                                 ::testing::ValuesIn(vecCpuConfigs),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         MaxPoolingV8LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolV8_CPU_5D_ref, MaxPoolingV8LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsMaxV85D_ref),
                                 ::testing::ValuesIn(inputShapes5D()),
                                 ::testing::ValuesIn((inpOutPrecision())),
                                 ::testing::Values(expectedCpuConfig()),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         MaxPoolingV8LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_5D, PoolingLayerCPUTest,
                         ::testing::Combine(
                              ::testing::ValuesIn(paramsAvg5D()),
                              ::testing::ValuesIn(inputShapes5D()),
                              ::testing::ValuesIn((inpOutPrecision())),
                              ::testing::Values(false),
                              ::testing::ValuesIn(vecCpuConfigs),
                              ::testing::Values(emptyFusingSpec),
                              ::testing::Values(cpuEmptyPluginConfig)),
                          PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_5D_NotOptimized, PoolingLayerCPUTest,
                         ::testing::Combine(
                              ::testing::ValuesIn(paramsAvg5D_RefOnly),
                              ::testing::ValuesIn(inputShapes5D()),
                              ::testing::ValuesIn((inpOutPrecision())),
                              ::testing::Values(false),
                              ::testing::Values(expectedCpuConfig()),
                              ::testing::Values(emptyFusingSpec),
                              ::testing::Values(cpuEmptyPluginConfig)),
                          PoolingLayerCPUTest::getTestCaseName);
#endif
} // namespace Pooling
} // namespace CPULayerTestsDefinitions
