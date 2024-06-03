// Copyright (C) 2023 Intel Corporation
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

const std::vector<CPUSpecificParams> vecCpuConfigs = {expectedCpuConfigAnyLayout()};

const std::vector<poolSpecificParams> paramsAvg3D_RefOnly = {
        poolSpecificParams{ ov::test::utils::PoolingTypes::AVG, {2}, {2}, {2}, {2},
                            expectedAvgRoundingType(), ov::op::PadType::EXPLICIT, false },
};

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool_CPU_3D, PoolingLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsMax3D()),
                                 ::testing::ValuesIn(inputShapes3D()),
                                 ::testing::ValuesIn((inpOutPrecision())),
                                 ::testing::Values(false),
                                 ::testing::ValuesIn(vecCpuConfigs),
                                 ::testing::Values(emptyFusingSpec)),
                         PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_3D, PoolingLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsAvg3D()),
                                 ::testing::ValuesIn(inputShapes3D()),
                                 ::testing::ValuesIn((inpOutPrecision())),
                                 ::testing::Values(false),
                                 ::testing::ValuesIn(vecCpuConfigs),
                                 ::testing::Values(emptyFusingSpec)),
                         PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_3D_NotOptimized, PoolingLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsAvg3D_RefOnly),
                                 ::testing::ValuesIn(inputShapes3D()),
                                 ::testing::ValuesIn((inpOutPrecision())),
                                 ::testing::Values(false),
                                 ::testing::Values(expectedCpuConfigAnyLayout()),
                                 ::testing::Values(emptyFusingSpec)),
                         PoolingLayerCPUTest::getTestCaseName);

const std::vector<poolSpecificParams> paramsAvg4D_RefOnly = {
        poolSpecificParams{ ov::test::utils::PoolingTypes::AVG, {2, 2}, {2, 2}, {2, 2}, {2, 2},
                            expectedAvgRoundingType(), ov::op::PadType::EXPLICIT, false },
};

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool_CPU_4D, PoolingLayerCPUTest,
                            ::testing::Combine(
                            ::testing::ValuesIn(paramsMax4D()),
                            ::testing::ValuesIn(inputShapes4D()),
                            ::testing::ValuesIn((inpOutPrecision())),
                            ::testing::Values(false),
                            ::testing::ValuesIn(filterCPUInfo(vecCpuConfigsFusing_4D())),
                            ::testing::Values(emptyFusingSpec)),
                        PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolV8_CPU_4D, MaxPoolingV8LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsMaxV84D()),
                                 ::testing::ValuesIn(inputShapes4D()),
                                 ::testing::ValuesIn((inpOutPrecision())),
                                 ::testing::ValuesIn(filterCPUInfo(vecCpuConfigsFusing_4D()))),
                         MaxPoolingV8LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_4D, PoolingLayerCPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(paramsAvg4D()),
                            ::testing::ValuesIn(inputShapes4D()),
                            ::testing::ValuesIn((inpOutPrecision())),
                            ::testing::Values(false),
                            ::testing::ValuesIn(vecCpuConfigs),
                            ::testing::Values(emptyFusingSpec)),
                        PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_4D_NotOptimized, PoolingLayerCPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(paramsAvg4D_RefOnly),
                            ::testing::ValuesIn(inputShapes4D()),
                            ::testing::ValuesIn((inpOutPrecision())),
                            ::testing::Values(false),
                            ::testing::Values(expectedCpuConfigAnyLayout()),
                            ::testing::Values(emptyFusingSpec)),
                        PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_Large, PoolingLayerCPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(paramsAvg4D_Large()),
                            ::testing::ValuesIn(inputShapes4D_Large()),
                            ::testing::ValuesIn((inpOutPrecision())),
                            ::testing::Values(false),
                            ::testing::ValuesIn(vecCpuConfigs),
                            ::testing::Values(emptyFusingSpec)),
                        PoolingLayerCPUTest::getTestCaseName);

const std::vector<maxPoolV8SpecificParams> paramsMaxV85D_ref = {
        maxPoolV8SpecificParams{ {2, 2, 2}, {1, 1, 1}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0},
                                                        ov::element::Type_t::i32, 0,
                                                        ov::op::RoundingType::CEIL, ov::op::PadType::SAME_UPPER },
        maxPoolV8SpecificParams{ {2, 2, 2}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1},
                                                        ov::element::Type_t::i32, 0,
                                                        ov::op::RoundingType::CEIL, ov::op::PadType::EXPLICIT },
        maxPoolV8SpecificParams{ {2, 3, 4}, {2, 2, 2}, {2, 1, 1}, {1, 1, 1}, {1, 2, 2},
                                                        ov::element::Type_t::i32, 0,
                                                        ov::op::RoundingType::CEIL, ov::op::PadType::EXPLICIT },
};

const std::vector<poolSpecificParams> paramsAvg5D_RefOnly = {
        poolSpecificParams{ ov::test::utils::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2},
                            expectedAvgRoundingType(), ov::op::PadType::EXPLICIT, false },
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
                             ::testing::Values(emptyFusingSpec)),
                         PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolV8_CPU_5D, MaxPoolingV8LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsMaxV85D()),
                                 ::testing::ValuesIn(inputShapes5D()),
                                 ::testing::ValuesIn((inpOutPrecision())),
                                 ::testing::ValuesIn(vecCpuConfigs)),
                         MaxPoolingV8LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolV8_CPU_5D_ref, MaxPoolingV8LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsMaxV85D_ref),
                                 ::testing::ValuesIn(inputShapes5D()),
                                 ::testing::ValuesIn((inpOutPrecision())),
                                 ::testing::Values(expectedCpuConfigAnyLayout())),
                         MaxPoolingV8LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_5D, PoolingLayerCPUTest,
                         ::testing::Combine(
                              ::testing::ValuesIn(paramsAvg5D()),
                              ::testing::ValuesIn(inputShapes5D()),
                              ::testing::ValuesIn((inpOutPrecision())),
                              ::testing::Values(false),
                              ::testing::ValuesIn(vecCpuConfigs),
                              ::testing::Values(emptyFusingSpec)),
                          PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_5D_NotOptimized, PoolingLayerCPUTest,
                         ::testing::Combine(
                              ::testing::ValuesIn(paramsAvg5D_RefOnly),
                              ::testing::ValuesIn(inputShapes5D()),
                              ::testing::ValuesIn((inpOutPrecision())),
                              ::testing::Values(false),
                              ::testing::Values(expectedCpuConfigAnyLayout()),
                              ::testing::Values(emptyFusingSpec)),
                          PoolingLayerCPUTest::getTestCaseName);
#endif
}  // namespace Pooling
}  // namespace test
}  // namespace ov
