// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/grid_sample.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using namespace GridSample;

namespace {

std::vector<CPUSpecificParams> getCPUInfoForARM() {
    std::vector<CPUSpecificParams> resCPUParams;
    // ARM uses decomposition transformation, so we use "ref" executor
    resCPUParams.push_back(CPUSpecificParams{{}, {}, {"ref"}, "ref"});
    return resCPUParams;
}

// ARM-specific shapes for testing
const std::vector<std::vector<InputShape>> staticShapesARM = {
    {{{}, {{1, 3, 5, 5}}}, {{}, {{1, 5, 5, 2}}}},
    {{{}, {{2, 2, 10, 10}}}, {{}, {{2, 8, 8, 2}}}},
    {{{}, {{1, 1, 20, 20}}}, {{}, {{1, 15, 15, 2}}}},
    {{{}, {{3, 3, 30, 30}}}, {{}, {{3, 25, 25, 2}}}},
    // RT-DETR-like shapes
    {{{}, {{8, 32, 80, 80}}}, {{}, {{8, 300, 4, 2}}}},
    // Channel variations
    {{{}, {{1, 16, 4, 4}}}, {{}, {{1, 3, 3, 2}}}},
    {{{}, {{2, 64, 8, 8}}}, {{}, {{2, 4, 4, 2}}}},
};

// Dynamic shapes specific to ARM
const std::vector<std::vector<InputShape>> dynamicShapesARM = {
    {{{{1, 5}, 3, {5, 50}, {5, 50}},
      {{1, 3, 10, 10}, {2, 3, 20, 20}, {5, 3, 50, 50}}},
     {{{1, 5}, {5, 50}, {5, 50}, 2},
      {{1, 10, 10, 2}, {2, 20, 20, 2}, {5, 50, 50, 2}}}},
    {{{-1, -1, -1, -1},
      {{1, 4, 16, 16}, {2, 8, 32, 32}}},
     {{-1, -1, -1, 2},
      {{1, 12, 12, 2}, {2, 24, 24, 2}}}}
};

}  // namespace

// Static shapes tests for ARM - using armInterpolationModes (BILINEAR + NEAREST only)
INSTANTIATE_TEST_SUITE_P(smoke_GridSample_ARM_Static, GridSampleLayerTestCPU,
                        ::testing::Combine(
                            ::testing::ValuesIn(staticShapesARM),
                            ::testing::ValuesIn(armInterpolationModes()),
                            ::testing::ValuesIn(allPaddingModes()),
                            ::testing::ValuesIn(alignCornersValues()),
                            ::testing::Values(ElementType::f32),
                            ::testing::Values(ElementType::f32),
                            ::testing::ValuesIn(getCPUInfoForARM()),
                            ::testing::Values(ov::AnyMap{})),
                        GridSampleLayerTestCPU::getTestCaseName);

// Dynamic shapes tests for ARM - using armInterpolationModes (BILINEAR + NEAREST only)
INSTANTIATE_TEST_SUITE_P(smoke_GridSample_ARM_Dynamic, GridSampleLayerTestCPU,
                        ::testing::Combine(
                            ::testing::ValuesIn(dynamicShapesARM),
                            ::testing::ValuesIn(armInterpolationModes()),
                            ::testing::ValuesIn(allPaddingModes()),
                            ::testing::ValuesIn(alignCornersValues()),
                            ::testing::Values(ElementType::f32),
                            ::testing::Values(ElementType::f32),
                            ::testing::ValuesIn(getCPUInfoForARM()),
                            ::testing::Values(ov::AnyMap{})),
                        GridSampleLayerTestCPU::getTestCaseName);

// FP16 tests for ARM (common on ARM platforms)
INSTANTIATE_TEST_SUITE_P(smoke_GridSample_ARM_FP16, GridSampleLayerTestCPU,
                        ::testing::Combine(
                            ::testing::Values(staticShapesARM[0]),
                            ::testing::Values(ov::op::v9::GridSample::InterpolationMode::BILINEAR),
                            ::testing::Values(ov::op::v9::GridSample::PaddingMode::ZEROS),
                            ::testing::Values(false),
                            ::testing::Values(ElementType::f16),
                            ::testing::Values(ElementType::f16),
                            ::testing::ValuesIn(getCPUInfoForARM()),
                            ::testing::Values(ov::AnyMap{})),
                        GridSampleLayerTestCPU::getTestCaseName);

// RT-DETR specific test
INSTANTIATE_TEST_SUITE_P(rtdetr_GridSample_ARM, GridSampleLayerTestCPU,
                        ::testing::Combine(
                            ::testing::Values(std::vector<InputShape>{
                                {{}, {{8, 32, 80, 80}}},
                                {{}, {{8, 300, 4, 2}}}}),
                            ::testing::Values(ov::op::v9::GridSample::InterpolationMode::BILINEAR),
                            ::testing::Values(ov::op::v9::GridSample::PaddingMode::ZEROS),
                            ::testing::Values(false),
                            ::testing::Values(ElementType::f16),
                            ::testing::Values(ElementType::f16),
                            ::testing::ValuesIn(getCPUInfoForARM()),
                            ::testing::Values(ov::AnyMap{{ov::hint::inference_precision.name(), ov::element::f16}})),
                        GridSampleLayerTestCPU::getTestCaseName);

}  // namespace test
}  // namespace ov