// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/grid_sample.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using namespace GridSample;

// Static shapes tests - same as original master
INSTANTIATE_TEST_SUITE_P(smoke_GridSample_Static,
                        GridSampleLayerTestCPU,
                        ::testing::Combine(::testing::ValuesIn(getStaticShapes()),
                                           ::testing::ValuesIn(allInterpolationModes()),
                                           ::testing::ValuesIn(allPaddingModes()),
                                           ::testing::ValuesIn(alignCornersValues()),
                                           ::testing::ValuesIn({ElementType::f32, ElementType::i32}),
                                           ::testing::ValuesIn({ElementType::f32}),
                                           ::testing::ValuesIn(getCPUInfoForCommon()),
                                           ::testing::Values(additionalConfigs()[0])),
                        GridSampleLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_GridSample_Static_1,
                        GridSampleLayerTestCPU,
                        ::testing::Combine(::testing::ValuesIn(getStaticShapes()),
                                           ::testing::ValuesIn(allInterpolationModes()),
                                           ::testing::ValuesIn(allPaddingModes()),
                                           ::testing::ValuesIn(alignCornersValues()),
                                           ::testing::ValuesIn({ElementType::bf16, ElementType::i8}),
                                           ::testing::ValuesIn({ElementType::f32, ElementType::bf16}),
                                           ::testing::ValuesIn(getCPUInfoForCommon()),
                                           ::testing::Values(additionalConfigs()[0])),
                        GridSampleLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_GridSample_Static_2,
                        GridSampleLayerTestCPU,
                        ::testing::Combine(::testing::ValuesIn(getStaticShapes()),
                                           ::testing::ValuesIn(allInterpolationModes()),
                                           ::testing::ValuesIn(allPaddingModes()),
                                           ::testing::ValuesIn(alignCornersValues()),
                                           ::testing::ValuesIn({ElementType::f32}),
                                           ::testing::ValuesIn({ElementType::bf16}),
                                           ::testing::ValuesIn(getCPUInfoForCommon()),
                                           ::testing::Values(additionalConfigs()[0])),
                        GridSampleLayerTestCPU::getTestCaseName);

// Dynamic shapes tests - same as original master
INSTANTIATE_TEST_SUITE_P(smoke_GridSample_Dynamic,
                        GridSampleLayerTestCPU,
                        ::testing::Combine(::testing::ValuesIn(getDynamicShapes()),
                                           ::testing::ValuesIn(allInterpolationModes()),
                                           ::testing::ValuesIn(allPaddingModes()),
                                           ::testing::ValuesIn(alignCornersValues()),
                                           ::testing::Values(ElementType::f32),
                                           ::testing::Values(ElementType::f32),
                                           ::testing::ValuesIn(getCPUInfoForCommon()),
                                           ::testing::Values(additionalConfigs()[0])),
                        GridSampleLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_GridSample_Dynamic,
                        GridSampleLayerTestCPU,
                        ::testing::Combine(::testing::ValuesIn(getDynamicShapes()),
                                           ::testing::ValuesIn(allInterpolationModes()),
                                           ::testing::ValuesIn(allPaddingModes()),
                                           ::testing::ValuesIn(alignCornersValues()),
                                           ::testing::ValuesIn({ElementType::bf16, ElementType::i32}),
                                           ::testing::ValuesIn({ElementType::bf16}),
                                           ::testing::ValuesIn(getCPUInfoForCommon()),
                                           ::testing::Values(additionalConfigs()[0])),
                        GridSampleLayerTestCPU::getTestCaseName);

}  // namespace test
}  // namespace ov