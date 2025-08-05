// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/grid_sample.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov::test {

using namespace GridSample;

namespace {

std::vector<ov::AnyMap> additionalConfigX64 = {{{ov::hint::inference_precision(ov::element::f32)}},
                                               {{ov::hint::inference_precision(ov::element::bf16)}}};

std::vector<CPUSpecificParams> getCPUInfoX64() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (ov::with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_avx512"}, "jit_avx512"});
    } else if (ov::with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_avx2"}, "jit_avx2"});
    } else if (ov::with_cpu_x86_avx()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_avx"}, "jit_avx"});
    } else if (ov::with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_sse42"}, "jit_sse42"});
    }
    return resCPUParams;
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(x64_smoke_static,
                         GridSampleLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(getStaticShapes()),
                                            ::testing::ValuesIn(allInterpolationModes()),
                                            ::testing::ValuesIn(allPaddingModes()),
                                            ::testing::ValuesIn(alignCornersValues()),
                                            ::testing::ValuesIn({ElementType::f32, ElementType::i32}),
                                            ::testing::ValuesIn({ElementType::f32}),
                                            ::testing::ValuesIn(getCPUInfoX64()),
                                            ::testing::Values(additionalConfigX64[0])),
                         GridSampleLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(x64_nightly_static_1,
                         GridSampleLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(getStaticShapes()),
                                            ::testing::ValuesIn(allInterpolationModes()),
                                            ::testing::ValuesIn(allPaddingModes()),
                                            ::testing::ValuesIn(alignCornersValues()),
                                            ::testing::ValuesIn({ElementType::bf16, ElementType::i8}),
                                            ::testing::ValuesIn({ElementType::f32, ElementType::bf16}),
                                            ::testing::ValuesIn(getCPUInfoX64()),
                                            ::testing::Values(additionalConfigX64[0])),
                         GridSampleLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(x64_nightly_static_2,
                         GridSampleLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(getStaticShapes()),
                                            ::testing::ValuesIn(allInterpolationModes()),
                                            ::testing::ValuesIn(allPaddingModes()),
                                            ::testing::ValuesIn(alignCornersValues()),
                                            ::testing::ValuesIn({ElementType::f32}),
                                            ::testing::ValuesIn({ElementType::bf16}),
                                            ::testing::ValuesIn(getCPUInfoX64()),
                                            ::testing::Values(additionalConfigX64[0])),
                         GridSampleLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(x64_smoke_dynamic,
                         GridSampleLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(getDynamicShapes()),
                                            ::testing::ValuesIn(allInterpolationModes()),
                                            ::testing::ValuesIn(allPaddingModes()),
                                            ::testing::ValuesIn(alignCornersValues()),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(getCPUInfoX64()),
                                            ::testing::Values(additionalConfigX64[0])),
                         GridSampleLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(x64_nightly_dynamic,
                         GridSampleLayerTestCPU,
                         ::testing::Combine(::testing::ValuesIn(getDynamicShapes()),
                                            ::testing::ValuesIn(allInterpolationModes()),
                                            ::testing::ValuesIn(allPaddingModes()),
                                            ::testing::ValuesIn(alignCornersValues()),
                                            ::testing::ValuesIn({ElementType::bf16, ElementType::i32}),
                                            ::testing::ValuesIn({ElementType::bf16}),
                                            ::testing::ValuesIn(getCPUInfoX64()),
                                            ::testing::Values(additionalConfigX64[0])),
                         GridSampleLayerTestCPU::getTestCaseName);

}  // namespace ov::test
