// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/grid_sample.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using namespace GridSample;

// Static shapes tests
INSTANTIATE_TEST_SUITE_P(smoke_GridSample_Static, GridSampleLayerTestCPU,
                        ::testing::Combine(
                            ::testing::ValuesIn(getStaticShapes()),
                            ::testing::ValuesIn(allInterpolationModes()),
                            ::testing::ValuesIn(allPaddingModes()),
                            ::testing::ValuesIn(alignCornersValues()),
                            ::testing::Values(ElementType::f32),
                            ::testing::Values(ElementType::f32),
                            ::testing::ValuesIn(getCPUInfoForCommon()),
                            ::testing::Values(ov::AnyMap{})),
                        GridSampleLayerTestCPU::getTestCaseName);

// Dynamic shapes tests
INSTANTIATE_TEST_SUITE_P(smoke_GridSample_Dynamic, GridSampleLayerTestCPU,
                        ::testing::Combine(
                            ::testing::ValuesIn(getDynamicShapes()),
                            ::testing::ValuesIn(allInterpolationModes()),
                            ::testing::ValuesIn(allPaddingModes()),
                            ::testing::ValuesIn(alignCornersValues()),
                            ::testing::Values(ElementType::f32),
                            ::testing::Values(ElementType::f32),
                            ::testing::ValuesIn(getCPUInfoForCommon()),
                            ::testing::Values(ov::AnyMap{})),
                        GridSampleLayerTestCPU::getTestCaseName);

// BF16 precision tests
INSTANTIATE_TEST_SUITE_P(smoke_GridSample_BF16, GridSampleLayerTestCPU,
                        ::testing::Combine(
                            ::testing::Values(getStaticShapes()[0]),
                            ::testing::ValuesIn(allInterpolationModes()),
                            ::testing::ValuesIn(allPaddingModes()),
                            ::testing::ValuesIn(alignCornersValues()),
                            ::testing::Values(ElementType::f32),
                            ::testing::Values(ElementType::bf16),
                            ::testing::ValuesIn(getCPUInfoForCommon()),
                            ::testing::ValuesIn(additionalConfigs())),
                        GridSampleLayerTestCPU::getTestCaseName);

}  // namespace test
}  // namespace ov