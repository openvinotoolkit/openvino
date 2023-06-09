// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/pooling.hpp"
#include "shared_test_classes/single_layer/pooling.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include <ngraph_functions/builders.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;


namespace CPULayerTestsDefinitions {
namespace Pooling {
namespace {

const auto avx512 = CPUSpecificParams{{}, {}, {"jit_avx512"}, "jit_avx512"};
const auto avx = CPUSpecificParams{{}, {}, {"jit_avx"}, "jit_avx"};
const auto sse42 = CPUSpecificParams{{}, {}, {"jit_sse42"}, "jit_sse42"};

const std::vector<CPUSpecificParams> vecCpuConfigs = {sse42, avx, avx512};

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

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_4D, PoolingLayerCPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(paramsAvg4D),
                            ::testing::ValuesIn(inputShapes4D),
                            ::testing::ValuesIn(inpOutPrecision),
                            ::testing::Values(false),
                            ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs)),
                            ::testing::Values(emptyFusingSpec)),
                        PoolingLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_Large, PoolingLayerCPUTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(paramsAvg4D_Large),
                            ::testing::ValuesIn(inputShapes4D_Large),
                            ::testing::ValuesIn(inpOutPrecision),
                            ::testing::Values(false),
                            ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs)),
                            ::testing::Values(emptyFusingSpec)),
                        PoolingLayerCPUTest::getTestCaseName);

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

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_5D, PoolingLayerCPUTest,
                         ::testing::Combine(
                              ::testing::ValuesIn(paramsAvg5D),
                              ::testing::ValuesIn(inputShapes5D),
                              ::testing::ValuesIn(inpOutPrecision),
                              ::testing::Values(false),
                              ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigs)),
                              ::testing::Values(emptyFusingSpec)),
                          PoolingLayerCPUTest::getTestCaseName);

/* === Fusing === */

const auto avx512_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {"jit_avx512"}, "jit_avx512"};
const auto avx512_ndhwc = CPUSpecificParams{{ndhwc}, {ndhwc}, {"jit_avx512"}, "jit_avx512"};

const auto avx2_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {"jit_avx2"}, "jit_avx2"};
const auto avx2_ndhwc = CPUSpecificParams{{ndhwc}, {ndhwc}, {"jit_avx2"}, "jit_avx2"};

const auto sse42_nhwc = CPUSpecificParams{{nhwc}, {nhwc}, {"jit_sse42"}, "jit_sse42"};
const auto sse42_ndhwc = CPUSpecificParams{{ndhwc}, {ndhwc}, {"jit_sse42"}, "jit_sse42"};

const std::vector<CPUSpecificParams> vecCpuConfigsFusing_4D = {sse42_nhwc, avx2_nhwc, avx512_nhwc};
const std::vector<CPUSpecificParams> vecCpuConfigsFusing_5D = {sse42_ndhwc, avx2_ndhwc, avx512_ndhwc};

std::vector<fusingSpecificParams> fusingParamsSet {
    emptyFusingSpec,
    fusingFakeQuantizePerTensor,
    fusingFakeQuantizePerChannel,
};

const std::vector<InputShape> inputShapes4D_int8 = {
        { {}, {{3, 4, 64, 64}} },
        { {}, {{2, 8, 8, 12}} },
        { {}, {{1, 16, 16, 12}} },
        { {}, {{1, 21, 8, 4}} },
        { {}, {{1, 32, 8, 8}} },
        {
            // dynamic
            {-1, 32, -1, -1},
            // target
            {
                {1, 32, 8, 8},
                {1, 32, 8, 4},
                {2, 32, 8, 12},
                {1, 32, 8, 8}
            }
        },
        {
            // dynamic
            {{1, 5}, 16, {1, 64}, {1, 64}},
            // target
            {
                {3, 16, 32, 32},
                {1, 16, 16, 12},
                {1, 16, 8, 8},
                {3, 16, 32, 32},
            }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_4D_I8, PoolingLayerCPUTest,
                         ::testing::Combine(
                              ::testing::ValuesIn(paramsAvg4D),
                              ::testing::ValuesIn(inputShapes4D_int8),
                              ::testing::Values(ElementType::f32),
                              ::testing::Values(true),
                              ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigsFusing_4D)),
                              ::testing::ValuesIn(fusingParamsSet)),
                          PoolingLayerCPUTest::getTestCaseName);

const std::vector<InputShape> inputShapes5D_int8 = {
        { {}, {{1, 4, 16, 16, 16}} },
        { {}, {{2, 8, 8, 8, 8}} },
        { {}, {{2, 16, 12, 16, 20}} },
        { {}, {{1, 19, 16, 20, 8}} },
        { {}, {{1, 32, 16, 8, 12}} },
        {
            // dynamic
            {-1, 32, -1, -1, -1},
            // target
            {
                {2, 32, 8, 8, 8},
                {1, 32, 16, 20, 8},
                {1, 32, 16, 16, 16},
                {2, 32, 8, 8, 8}
            }
        },
        {
            // dynamic
            {{1, 5}, 16, {1, 64}, {1, 64}, {1, 25}},
            // target
            {
                {1, 16, 16, 16, 16},
                {1, 16, 16, 8, 12},
                {2, 16, 8, 8, 8},
                {1, 16, 16, 16, 16},
            }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_AvgPool_CPU_5D_I8, PoolingLayerCPUTest,
                         ::testing::Combine(
                              ::testing::ValuesIn(paramsAvg5D),
                              ::testing::ValuesIn(inputShapes5D_int8),
                              ::testing::Values(ElementType::f32),
                              ::testing::Values(true),
                              ::testing::ValuesIn(filterCPUInfoForDevice(vecCpuConfigsFusing_5D)),
                              ::testing::ValuesIn(fusingParamsSet)),
                          PoolingLayerCPUTest::getTestCaseName);
} // namespace
} // namespace Pooling
} // namespace CPULayerTestsDefinitions