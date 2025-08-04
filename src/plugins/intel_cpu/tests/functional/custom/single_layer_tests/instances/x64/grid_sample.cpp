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

std::vector<CPUSpecificParams> getCPUInfoForX64() {
    std::vector<CPUSpecificParams> resCPUParams;
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_avx512"}, "jit_avx512"});
    }
    if (with_cpu_x86_avx2()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_avx2"}, "jit_avx2"});
    }
    if (with_cpu_x86_sse42()) {
        resCPUParams.push_back(CPUSpecificParams{{}, {}, {"jit_sse42"}, "jit_sse42"});
    }
    resCPUParams.push_back(CPUSpecificParams{{}, {}, {"ref"}, "ref"});
    return resCPUParams;
}

}  // namespace

// X86 JIT optimized tests
INSTANTIATE_TEST_SUITE_P(smoke_GridSample_x86_JIT, GridSampleLayerTestCPU,
                        ::testing::Combine(
                            ::testing::Values(getStaticShapes()[0], getStaticShapes()[1]),
                            ::testing::Values(ov::op::v9::GridSample::InterpolationMode::BILINEAR),  // JIT optimized
                            ::testing::Values(ov::op::v9::GridSample::PaddingMode::ZEROS, 
                                              ov::op::v9::GridSample::PaddingMode::BORDER),
                            ::testing::Values(false),  // align_corners = false is more optimized
                            ::testing::Values(ElementType::f32),
                            ::testing::Values(ElementType::f32),
                            ::testing::ValuesIn(getCPUInfoForX64()),
                            ::testing::Values(ov::AnyMap{})),
                        GridSampleLayerTestCPU::getTestCaseName);

// X86 specific dynamic shapes
const std::vector<std::vector<InputShape>> dynamicShapesX86 = {
    {{{{1, 10}, {1, 10}, {1, 100}, {1, 100}},
      {{1, 1, 50, 50}, {3, 3, 100, 100}, {10, 10, 25, 25}}},
     {{{1, 10}, {1, 100}, {1, 100}, 2},
      {{1, 50, 50, 2}, {3, 100, 100, 2}, {10, 25, 25, 2}}}}
};

INSTANTIATE_TEST_SUITE_P(smoke_GridSample_x86_Dynamic, GridSampleLayerTestCPU,
                        ::testing::Combine(
                            ::testing::ValuesIn(dynamicShapesX86),
                            ::testing::Values(ov::op::v9::GridSample::InterpolationMode::BILINEAR),
                            ::testing::Values(ov::op::v9::GridSample::PaddingMode::ZEROS, 
                                              ov::op::v9::GridSample::PaddingMode::BORDER),
                            ::testing::Values(false),
                            ::testing::Values(ElementType::f32),
                            ::testing::Values(ElementType::f32),
                            ::testing::ValuesIn(getCPUInfoForX64()),
                            ::testing::Values(ov::AnyMap{})),
                        GridSampleLayerTestCPU::getTestCaseName);

}  // namespace test
}  // namespace ov