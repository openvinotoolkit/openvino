// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/grid_sample.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov::test {

using namespace GridSample;

namespace {

std::vector<ov::AnyMap> additionalConfigARM = {{{ov::hint::inference_precision(ov::element::f32)}},
                                               {{ov::hint::inference_precision(ov::element::f16)}}};

std::vector<CPUSpecificParams> getCPUInfoARM() {
    std::vector<CPUSpecificParams> resCPUParams;
    resCPUParams.push_back(CPUSpecificParams{{}, {}, {"ref"}, "ref"});
    return resCPUParams;
}

}  // namespace

// Keep ARM coverage at parity with master monolithic test
static std::vector<std::vector<InputShape>> getStaticShapesARM() {
    // Base SSE-like set
    std::vector<std::vector<InputShape>> result = {
        {{{}, {{1, 5, 1, 1}}}, {{}, {{1, 1, 1, 2}}}},
        {{{}, {{2, 4, 7, 1}}}, {{}, {{2, 1, 2, 2}}}},
        {{{}, {{3, 3, 3, 3}}}, {{}, {{3, 3, 1, 2}}}},
        {{{}, {{4, 2, 5, 4}}}, {{}, {{4, 2, 2, 2}}}},
        {{{}, {{5, 1, 5, 5}}}, {{}, {{5, 1, 5, 2}}}},
        {{{}, {{4, 2, 4, 6}}}, {{}, {{4, 2, 3, 2}}}},
        {{{}, {{3, 3, 5, 7}}}, {{}, {{3, 7, 1, 2}}}},
        {{{}, {{2, 4, 7, 7}}}, {{}, {{2, 2, 4, 2}}}},
        {{{}, {{2, 5, 8, 8}}}, {{}, {{2, 3, 3, 2}}}},
        {{{}, {{2, 6, 9, 8}}}, {{}, {{2, 2, 5, 2}}}},
    };
    // Extend like AVX/AVX2 set in master to keep parity
    std::vector<std::vector<InputShape>> extra = {
        {{{}, {{1, 7, 5, 3}}}, {{}, {{1, 1, 11, 2}}}},
        {{{}, {{2, 6, 7, 2}}}, {{}, {{2, 6, 2, 2}}}},
        {{{}, {{3, 2, 9, 1}}}, {{}, {{3, 3, 13, 2}}}},
        {{{}, {{4, 7, 3, 4}}}, {{}, {{4, 5, 5, 2}}}},
        {{{}, {{5, 3, 2, 13}}}, {{}, {{5, 1, 31, 2}}}},
        {{{}, {{4, 3, 5, 14}}}, {{}, {{4, 4, 8, 2}}}},
        {{{}, {{3, 2, 2, 15}}}, {{}, {{3, 33, 1, 2}}}},
        {{{}, {{2, 1, 6, 16}}}, {{}, {{2, 8, 8, 2}}}},
        {{{}, {{2, 3, 7, 17}}}, {{}, {{2, 9, 9, 2}}}},
    };
    result.insert(result.end(), extra.begin(), extra.end());
    return result;
}

static const std::vector<std::vector<InputShape>> dynamicInShapesARM = {
    // from master dynamicInSapes
    {{{ov::Dimension(1, 15), -1, -1, -1},
      {{1, 1, 1, 1}, {6, 3, 1, 2}, {4, 5, 3, 1}, {2, 7, 2, 2}}},
     {{ov::Dimension(1, 16), -1, -1, -1},
      {{1, 1, 1, 2}, {6, 2, 2, 2}, {4, 1, 3, 2}, {2, 1, 2, 2}}}},
    {{{-1, -1, -1, -1},
      {{1, 2, 1, 5}, {3, 4, 2, 3}, {5, 6, 7, 1}, {7, 8, 2, 4}}},
     {{-1, -1, -1, 2},
      {{1, 2, 4, 2}, {3, 1, 7, 2}, {5, 2, 3, 2}, {7, 1, 5, 2}}}},
    {{{ov::Dimension(2, 15), -1, -1, -1},
      {{8, 3, 3, 3}, {6, 5, 2, 5}, {4, 7, 1, 11}, {2, 9, 3, 4}}},
     {{-1, 3, 7, 2},
      {{8, 3, 7, 2}, {6, 3, 7, 2}, {4, 3, 7, 2}, {2, 3, 7, 2}}}},
    {{{3, 4, 4, 5},
      {{3, 4, 4, 5}, {3, 4, 4, 5}, {3, 4, 4, 5}, {3, 4, 4, 5}}},
     {{-1, -1, -1, 2},
      {{3, 3, 4, 2}, {3, 1, 11, 2}, {3, 2, 5, 2}, {3, 3, 3, 2}}}},
    {{{-1, -1, -1, -1},
      {{1, 2, 1, 13}, {3, 4, 7, 2}, {5, 6, 3, 5}, {7, 8, 4, 4}}},
     {{-1, -1, -1, -1},
      {{1, 4, 4, 2}, {3, 3, 5, 2}, {5, 2, 7, 2}, {7, 1, 13, 2}}}},
    {{{-1, -1, -1, -1},
      {{2, 11, 1, 17}, {4, 9, 6, 3}, {6, 7, 7, 3}, {8, 3, 2, 11}}},
     {{-1, -1, -1, 2},
      {{2, 5, 4, 2}, {4, 1, 19, 2}, {6, 6, 3, 2}, {8, 1, 17, 2}}}},
    {{{3, -1, -1, -1},
      {{3, 2, 1, 23}, {3, 4, 3, 8}, {3, 6, 5, 5}, {3, 8, 31, 1}}},
     {{-1, -1, -1, 2},
      {{3, 31, 1, 2}, {3, 6, 4, 2}, {3, 23, 1, 2}, {3, 11, 2, 2}}}},
    {{{-1, 3, -1, -1},
      {{8, 3, 8, 4}, {6, 3, 33, 1}, {4, 3, 8, 6}, {2, 3, 8, 8}}},
     {{-1, -1, -1, 2},
      {{8, 8, 8, 2}, {6, 8, 7, 2}, {4, 1, 33, 2}, {2, 4, 8, 2}}}},
};

INSTANTIATE_TEST_SUITE_P(ARM_smoke_static,
                         GridSampleLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(getStaticShapesARM()),
                                            ::testing::ValuesIn(allInterpolationModes()),
                                            ::testing::ValuesIn(allPaddingModes()),
                                            ::testing::ValuesIn(alignCornersValues()),
                                            ::testing::ValuesIn({ElementType::f32, ElementType::i32}),
                                            ::testing::ValuesIn({ElementType::f32}),
                                            ::testing::ValuesIn(getCPUInfoARM()),
                                            ::testing::Values(additionalConfigARM[0])),
                         GridSampleLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ARM_nightly_static_1,
                         GridSampleLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(getStaticShapesARM()),
                                            ::testing::ValuesIn(allInterpolationModes()),
                                            ::testing::ValuesIn(allPaddingModes()),
                                            ::testing::ValuesIn(alignCornersValues()),
                                            ::testing::ValuesIn({ElementType::f16, ElementType::i8}),
                                            ::testing::ValuesIn({ElementType::f32, ElementType::f16}),
                                            ::testing::ValuesIn(getCPUInfoARM()),
                                            ::testing::Values(additionalConfigARM[0])),
                         GridSampleLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ARM_nightly_static_2,
                         GridSampleLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(getStaticShapesARM()),
                                            ::testing::ValuesIn(allInterpolationModes()),
                                            ::testing::ValuesIn(allPaddingModes()),
                                            ::testing::ValuesIn(alignCornersValues()),
                                            ::testing::ValuesIn({ElementType::f32}),
                                            ::testing::ValuesIn({ElementType::f16}),
                                            ::testing::ValuesIn(getCPUInfoARM()),
                                            ::testing::Values(additionalConfigARM[0])),
                         GridSampleLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ARM_smoke_dynamic,
                         GridSampleLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(dynamicInShapesARM),
                                            ::testing::ValuesIn(allInterpolationModes()),
                                            ::testing::ValuesIn(allPaddingModes()),
                                            ::testing::ValuesIn(alignCornersValues()),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(getCPUInfoARM()),
                                            ::testing::Values(additionalConfigARM[0])),
                         GridSampleLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ARM_nightly_dynamic,
                         GridSampleLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(dynamicInShapesARM),
                                            ::testing::ValuesIn(allInterpolationModes()),
                                            ::testing::ValuesIn(allPaddingModes()),
                                            ::testing::ValuesIn(alignCornersValues()),
                                            ::testing::ValuesIn({ElementType::f16, ElementType::i32}),
                                            ::testing::ValuesIn({ElementType::f16}),
                                            ::testing::ValuesIn(getCPUInfoARM()),
                                            ::testing::Values(additionalConfigARM[0])),
                         GridSampleLayerTestCPU::getTestCaseName);

}  // namespace ov::test
