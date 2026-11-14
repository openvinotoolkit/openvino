// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/rms_norm.hpp"

#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace RMSNorm {
const auto cpuSpec = CPUSpecificParams{{}, {}, {"ref_any"}, "ref_any"};
const auto jitAvx512Spec = []() -> std::vector<CPUSpecificParams> {
    return filterCPUInfoForDevice({CPUSpecificParams{{}, {}, {"jit"}, "jit_avx512"}});
};

const std::vector<std::vector<InputShape>> shapes{
    // normal
    {
        // data shape
        {ov::test::InputShape{ov::PartialShape{-1, -1, 1024 + 16 + 1},
                              {ov::Shape{1, 8, 1024 + 16 + 1}, ov::Shape{2, 3, 1024 + 16 + 1}}}},
        // scale shape
        {ov::test::InputShape{ov::PartialShape{1024 + 16 + 1}, {ov::Shape{1024 + 16 + 1}, ov::Shape{1024 + 16 + 1}}}},
    },
    // small data size
    {
        // data shape
        {ov::test::InputShape{ov::PartialShape{-1, -1, 31}, {ov::Shape{1, 8, 31}, ov::Shape{2, 3, 31}}}},
        // scale shape
        {ov::test::InputShape{ov::PartialShape{31}, {ov::Shape{31}, ov::Shape{31}}}},
    },
    // scale is scalar
    {
        // data shape
        {ov::test::InputShape{ov::PartialShape{-1, -1, 1094}, {ov::Shape{1, 8, 1094}, ov::Shape{2, 3, 1094}}}},
        // scale shape
        {ov::test::InputShape{ov::PartialShape{1}, {ov::Shape{1}, ov::Shape{1}}}},
    },
};

// Decomposition path (gain shape [1,8,64] is not a vector, so the RMS op is decomposed into primitives). The
// decomposition multiplies the gain and the normalized data without a Convert, so the gain must have the data
// precision here; the mixed data/gain precisions below only apply to shapes handled by the RMSNorm node.
const std::vector<std::vector<InputShape>> decomposedShapes{
    {
        // data shape
        {ov::test::InputShape{ov::PartialShape{-1, -1, 64}, {ov::Shape{1, 8, 64}}}},
        // scale shape
        {ov::test::InputShape{ov::PartialShape{1, 8, 64}, {ov::Shape{1, 8, 64}}}},
    },
};

const std::vector<std::vector<InputShape>> tinyRowBf16Shapes{
    {
        {ov::test::InputShape{ov::PartialShape{-1, -1, 15}, {ov::Shape{1, 8, 15}, ov::Shape{2, 3, 15}}}},
        {ov::test::InputShape{ov::PartialShape{15}, {ov::Shape{15}, ov::Shape{15}}}},
    },
    {
        {ov::test::InputShape{ov::PartialShape{-1, -1, 7}, {ov::Shape{1, 8, 7}, ov::Shape{2, 3, 7}}}},
        {ov::test::InputShape{ov::PartialShape{7}, {ov::Shape{7}, ov::Shape{7}}}},
    },
    {
        {ov::test::InputShape{ov::PartialShape{-1, -1, 8}, {ov::Shape{1, 8, 8}, ov::Shape{2, 3, 8}}}},
        {ov::test::InputShape{ov::PartialShape{8}, {ov::Shape{8}, ov::Shape{8}}}},
    },
    {
        {ov::test::InputShape{ov::PartialShape{-1, -1, 3}, {ov::Shape{1, 8, 3}, ov::Shape{2, 3, 3}}}},
        {ov::test::InputShape{ov::PartialShape{3}, {ov::Shape{3}, ov::Shape{3}}}},
    },
};

// RMS without gamma: data input only
const std::vector<std::vector<InputShape>> noScaleShapes{
    {
        // data shape
        {ov::test::InputShape{ov::PartialShape{-1, -1, 1024 + 16 + 1},
                              {ov::Shape{1, 8, 1024 + 16 + 1}, ov::Shape{2, 3, 1024 + 16 + 1}}}},
    },
    {
        // data shape
        {ov::test::InputShape{ov::PartialShape{-1, -1, 31}, {ov::Shape{1, 8, 31}, ov::Shape{2, 3, 31}}}},
    },
};

// The gain precision is an independent axis: same-precision and mixed combinations (e.g. a bf16 gain on f32 data,
// or an f32 gain on bf16 data) exercise the kernel's native-precision gain load as well as the f32 one.
const auto params = testing::Combine(testing::Values(ElementType::f32, ElementType::bf16, ElementType::f16),
                                     testing::Values(ElementType::f32, ElementType::bf16, ElementType::f16),
                                     testing::ValuesIn(shapes),
                                     testing::Values(ov::test::utils::DEVICE_CPU),
                                     testing::Values(cpuSpec));

const auto tinyRowBf16Params = testing::Combine(testing::Values(ElementType::bf16),
                                                testing::Values(ElementType::bf16, ElementType::f32),
                                                testing::ValuesIn(tinyRowBf16Shapes),
                                                testing::Values(ov::test::utils::DEVICE_CPU),
                                                testing::ValuesIn(jitAvx512Spec()));

INSTANTIATE_TEST_SUITE_P(smoke_RMSNorm_CPU, RMSNormLayerCPUTest, params, RMSNormLayerCPUTest::getTestCaseName);

// Gain precision equal to the data precision (a tied pair cannot be expressed with Combine).
static std::vector<RMSNormCPUTestParams> decomposedParams() {
    std::vector<RMSNormCPUTestParams> res;
    for (const auto prc : {ElementType::f32, ElementType::bf16, ElementType::f16}) {
        for (const auto& shape : decomposedShapes) {
            res.emplace_back(prc, prc, shape, ov::test::utils::DEVICE_CPU, cpuSpec);
        }
    }
    return res;
}

INSTANTIATE_TEST_SUITE_P(smoke_RMSNorm_Decomposed_CPU,
                         RMSNormLayerCPUTest,
                         testing::ValuesIn(decomposedParams()),
                         RMSNormLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_RMSNorm_TinyRow_BF16_CPU,
                         RMSNormLayerCPUTest,
                         tinyRowBf16Params,
                         RMSNormLayerCPUTest::getTestCaseName);

const auto noScaleParams = testing::Combine(testing::Values(ElementType::f32, ElementType::bf16, ElementType::f16),
                                            testing::Values(ElementType::f32),  // no scale input: unused
                                            testing::ValuesIn(noScaleShapes),
                                            testing::Values(ov::test::utils::DEVICE_CPU),
                                            testing::Values(cpuSpec));

INSTANTIATE_TEST_SUITE_P(smoke_RMSNorm_NoScale_CPU,
                         RMSNormLayerCPUTest,
                         noScaleParams,
                         RMSNormLayerCPUTest::getTestCaseName);

}  // namespace RMSNorm
}  // namespace test
}  // namespace ov
