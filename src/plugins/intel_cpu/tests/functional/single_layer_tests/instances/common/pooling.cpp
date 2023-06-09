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
const auto ref = CPUSpecificParams{{}, {}, {"ref_any"}, "ref_any"};

const std::vector<CPUSpecificParams> vecCpuConfigs = {ref};
const std::vector<ElementType> inpOutPrecision = {ElementType::f32/*, ElementType::bf16*/};

const std::vector<InputShape> inputShapes3D = {
        { {}, {{3, 4, 64}} },
        { {}, {{2, 8, 12}} },
        { {}, {{1, 16, 12}} },
        { {}, {{1, 21, 4}} },
        { {}, {{1, 32, 8}} },
        {
            // dynamic
            {-1, -1, -1},
            // target
            {
                {1, 32, 8},
                {1, 21, 4},
                {2, 8, 12}
            }
        },
        {
            // dynamic
            {{1, 5}, {4, 32}, {1, 64}},
            // target
            {
                {3, 4, 64},
                {1, 16, 12},
                {1, 32, 8}
            }
        }
};

const std::vector<InputShape> inputShapes4D = {
        { {}, {{3, 4, 64, 64}} },
        { {}, {{2, 8, 8, 12}} },
        { {}, {{1, 16, 16, 12}} },
        { {}, {{1, 21, 8, 4}} },
        { {}, {{1, 32, 8, 8}} },
        {
            // dynamic
            {-1, -1, -1, -1},
            // target
            {
                {1, 32, 8, 8},
                {1, 21, 8, 4},
                {2, 8, 8, 12},
                {1, 96, 125, 125}
            }
        },
        {
            // dynamic
            {{1, 5}, {4, 32}, {1, 64}, {1, 64}},
            // target
            {
                {3, 4, 64, 64},
                {1, 16, 16, 12},
                {1, 32, 8, 8}
            }
        },
        {
            // dynamic
            {{1, 10}, 16, 8, 8},
            // target
            {
                {1, 16, 8, 8},
                {2, 16, 8, 8},
            }
        }
};

const std::vector<InputShape> inputShapes5D = {
        { {}, {{1, 4, 16, 16, 16}} },
        { {}, {{2, 8, 8, 8, 8}} },
        { {}, {{2, 16, 12, 16, 20}} },
        { {}, {{1, 19, 16, 20, 8}} },
        { {}, {{1, 32, 16, 8, 12}} },
        {
            // dynamic
            {-1, -1, -1, -1, -1},
            // target
            {
                {2, 8, 8, 8, 8},
                {1, 19, 16, 20, 8},
                {1, 4, 16, 16, 16}
            }
        },
        {
            // dynamic
            {{1, 5}, {4, 32}, {1, 64}, {1, 64}, {1, 25}},
            // target
            {
                {1, 4, 16, 16, 16},
                {1, 32, 16, 8, 12},
                {3, 16, 4, 8, 3}
            }
        }
};

/* ============= Pooling (1D) ============= */
const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsMax3D = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2}, {2}, {0}, {0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {4}, {2}, {0}, {0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2}, {1}, {0}, {0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
};

const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg3D = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {3}, {1}, {1}, {0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {3}, {1}, {1}, {0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {4}, {4}, {2}, {2},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
};

const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg3D_RefOnly = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2}, {2}, {2}, {2},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
};

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool_CPU_3D, PoolingLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsMax3D),
                                 ::testing::ValuesIn(inputShapes3D),
                                 ::testing::ValuesIn(inpOutPrecision),
                                 ::testing::Values(false),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs)),
                                 ::testing::Values(emptyFusingSpec)),
                         PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_3D, PoolingLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsAvg3D),
                                 ::testing::ValuesIn(inputShapes3D),
                                 ::testing::ValuesIn(inpOutPrecision),
                                 ::testing::Values(false),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs)),
                                 ::testing::Values(emptyFusingSpec)),
                         PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_3D_NotOptimized, PoolingLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsAvg3D_RefOnly),
                                 ::testing::ValuesIn(inputShapes3D),
                                 ::testing::ValuesIn(inpOutPrecision),
                                 ::testing::Values(false),
                                 ::testing::Values(ref),
                                 ::testing::Values(emptyFusingSpec)),
                         PoolingLayerCPUTest::getTestCaseName);

/* ============= Pooling (2D) ============= */
const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsMax4D = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2}, {2, 2}, {0, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2}, {2, 2}, {0, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {4, 2}, {2, 2}, {0, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {4, 2}, {2, 1}, {0, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
};

const std::vector<LayerTestsDefinitions::maxPoolV8SpecificParams> paramsMaxV84D = {
        LayerTestsDefinitions::maxPoolV8SpecificParams{ {2, 2}, {2, 2}, {1, 1}, {0, 0}, {0, 0},
                                                        ngraph::element::Type_t::i32, 0,
                                                        ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER },
};

const std::vector<LayerTestsDefinitions::maxPoolV8SpecificParams> paramsMaxV84D_ref = {
        LayerTestsDefinitions::maxPoolV8SpecificParams{ {2, 2}, {2, 2}, {2, 2}, {0, 0}, {0, 0},
                                                        ngraph::element::Type_t::i32, 0,
                                                        ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER },
        LayerTestsDefinitions::maxPoolV8SpecificParams{ {4, 2}, {2, 2}, {1, 2}, {0, 0}, {0, 0},
                                                        ngraph::element::Type_t::i32, 0,
                                                        ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT },
        LayerTestsDefinitions::maxPoolV8SpecificParams{ {4, 2}, {2, 1}, {2, 2}, {0, 0}, {0, 0},
                                                        ngraph::element::Type_t::i32, 0,
                                                        ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT },
};

const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg4D = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {0, 0}, {0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {4, 4}, {4, 4}, {2, 2}, {2, 2},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
};

const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg4D_RefOnly = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2}, {2, 2}, {2, 2}, {2, 2},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
};

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool_CPU_4D, PoolingLayerCPUTest,
                            ::testing::Combine(
                            ::testing::ValuesIn(paramsMax4D),
                            ::testing::ValuesIn(inputShapes4D),
                            ::testing::ValuesIn(inpOutPrecision),
                            ::testing::Values(false),
                            ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs)),
                            ::testing::Values(emptyFusingSpec)),
                        PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolV8_CPU_4D, MaxPoolingV8LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsMaxV84D),
                                 ::testing::ValuesIn(inputShapes4D),
                                 ::testing::ValuesIn(inpOutPrecision),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs))),
                         MaxPoolingV8LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolV8_CPU_4D_ref, MaxPoolingV8LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsMaxV84D_ref),
                                 ::testing::ValuesIn(inputShapes4D),
                                 ::testing::ValuesIn(inpOutPrecision),
                                 ::testing::Values(ref)),
                         MaxPoolingV8LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_4D, PoolingLayerCPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(paramsAvg4D),
                            ::testing::ValuesIn(inputShapes4D),
                            ::testing::ValuesIn(inpOutPrecision),
                            ::testing::Values(false),
                            ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs)),
                            ::testing::Values(emptyFusingSpec)),
                        PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_4D_NotOptimized, PoolingLayerCPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(paramsAvg4D_RefOnly),
                            ::testing::ValuesIn(inputShapes4D),
                            ::testing::ValuesIn(inpOutPrecision),
                            ::testing::Values(false),
                            ::testing::Values(ref),
                            ::testing::Values(emptyFusingSpec)),
                        PoolingLayerCPUTest::getTestCaseName);

const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg4D_Large = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {65, 65}, {65, 65}, {0, 0}, {0, 0},
                            ngraph::op::RoundingType::FLOOR, ngraph::op::PadType::VALID, true },
};

const std::vector<InputShape> inputShapes4D_Large = {
        {
            // dynamic
            {-1, -1, -1, -1},
            // target
            {
                {1, 16, 65, 65},
                {1, 8, 130, 130},
                {1, 16, 65, 65}
            }
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_Large, PoolingLayerCPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(paramsAvg4D_Large),
                            ::testing::ValuesIn(inputShapes4D_Large),
                            ::testing::ValuesIn(inpOutPrecision),
                            ::testing::Values(false),
                            ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs)),
                            ::testing::Values(emptyFusingSpec)),
                        PoolingLayerCPUTest::getTestCaseName);

/* ============= Pooling (3D) ============= */
const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsMax5D = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::MAX, {3, 3, 3}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
};

const std::vector<LayerTestsDefinitions::maxPoolV8SpecificParams> paramsMaxV85D = {
        LayerTestsDefinitions::maxPoolV8SpecificParams{ {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0},
                                                        ngraph::element::Type_t::i32, 0,
                                                        ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER },
};

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

const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg5D = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {1, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {1, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {1, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_LOWER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {1, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::SAME_UPPER, false },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {3, 3, 3}, {3, 3, 3}, {1, 1, 1}, {0, 0, 0},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {4, 4, 4}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, true },
};

const std::vector<LayerTestsDefinitions::poolSpecificParams> paramsAvg5D_RefOnly = {
        LayerTestsDefinitions::poolSpecificParams{ ngraph::helpers::PoolingTypes::AVG, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2},
                            ngraph::op::RoundingType::CEIL, ngraph::op::PadType::EXPLICIT, false },
};

INSTANTIATE_TEST_SUITE_P(smoke_MaxPool_CPU_5D, PoolingLayerCPUTest,
                         ::testing::Combine(
                             ::testing::ValuesIn(paramsMax5D),
                             ::testing::ValuesIn(inputShapes5D),
                             ::testing::ValuesIn(inpOutPrecision),
                             ::testing::Values(false),
                             ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs)),
                             ::testing::Values(emptyFusingSpec)),
                         PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolV8_CPU_5D, MaxPoolingV8LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsMaxV85D),
                                 ::testing::ValuesIn(inputShapes5D),
                                 ::testing::ValuesIn(inpOutPrecision),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs))),
                         MaxPoolingV8LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolV8_CPU_5D_ref, MaxPoolingV8LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsMaxV85D_ref),
                                 ::testing::ValuesIn(inputShapes5D),
                                 ::testing::ValuesIn(inpOutPrecision),
                                 ::testing::Values(ref)),
                         MaxPoolingV8LayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_5D, PoolingLayerCPUTest,
                         ::testing::Combine(
                              ::testing::ValuesIn(paramsAvg5D),
                              ::testing::ValuesIn(inputShapes5D),
                              ::testing::ValuesIn(inpOutPrecision),
                              ::testing::Values(false),
                              ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs)),
                              ::testing::Values(emptyFusingSpec)),
                          PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_5D_NotOptimized, PoolingLayerCPUTest,
                         ::testing::Combine(
                              ::testing::ValuesIn(paramsAvg5D_RefOnly),
                              ::testing::ValuesIn(inputShapes5D),
                              ::testing::ValuesIn(inpOutPrecision),
                              ::testing::Values(false),
                              ::testing::Values(ref),
                              ::testing::Values(emptyFusingSpec)),
                          PoolingLayerCPUTest::getTestCaseName);
} // namespace Pooling
} // namespace CPULayerTestsDefinitions