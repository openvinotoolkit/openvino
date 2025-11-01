// Copyright (C) 2018-2025 Intel Corporation
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

INSTANTIATE_TEST_SUITE_P(ARM_smoke,
                         GridSampleLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(getAllShapes()),
                                            ::testing::ValuesIn(allInterpolationModes()),
                                            ::testing::ValuesIn(allPaddingModes()),
                                            ::testing::ValuesIn(alignCornersValues()),
                                            ::testing::ValuesIn({ElementType::f32, ElementType::i32}),
                                            ::testing::ValuesIn({ElementType::f32}),
                                            ::testing::ValuesIn(getCPUInfoARM()),
                                            ::testing::Values(additionalConfigARM[0])),
                         GridSampleLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ARM_nightly_1,
                         GridSampleLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(getAllShapes()),
                                            ::testing::ValuesIn(allInterpolationModes()),
                                            ::testing::ValuesIn(allPaddingModes()),
                                            ::testing::ValuesIn(alignCornersValues()),
                                            ::testing::ValuesIn({ElementType::f16, ElementType::i8}),
                                            ::testing::ValuesIn({ElementType::f32, ElementType::f16}),
                                            ::testing::ValuesIn(getCPUInfoARM()),
                                            ::testing::Values(additionalConfigARM[0])),
                         GridSampleLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ARM_nightly_2,
                         GridSampleLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(getAllShapes()),
                                            ::testing::ValuesIn(allInterpolationModes()),
                                            ::testing::ValuesIn(allPaddingModes()),
                                            ::testing::ValuesIn(alignCornersValues()),
                                            ::testing::ValuesIn({ElementType::f32, ElementType::i32}),
                                            ::testing::ValuesIn({ElementType::f16}),
                                            ::testing::ValuesIn(getCPUInfoARM()),
                                            ::testing::Values(additionalConfigARM[1])),
                         GridSampleLayerTestCPU::getTestCaseName);

}  // namespace ov::test
