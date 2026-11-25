// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/pooling.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Pooling {

// Alternate shapes whose last CEIL window starts inside and outside the trailing padding.
const std::vector<poolSpecificParams> paramsAvgV14CeilTorchDynamic = {
        {utils::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 1}, {1, 1},
         ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::EXPLICIT, false},
        {utils::PoolingTypes::AVG, {2, 2}, {2, 2}, {1, 1}, {1, 1},
         ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::EXPLICIT, true},
        {utils::PoolingTypes::AVG, {1, 1}, {2, 2}, {0, 0}, {0, 0},
         ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::EXPLICIT, true},
        {utils::PoolingTypes::AVG, {3, 2}, {2, 2}, {1, 1}, {1, 1},
         ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::EXPLICIT, true},
};

INSTANTIATE_TEST_SUITE_P(smoke_AvgPoolV14CeilTorchDynamic_CPU_4D, AvgPoolingV14LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsAvgV14CeilTorchDynamic),
                                 ::testing::Values(InputShape{{1, 3, -1, -1},
                                                             {{1, 3, 4, 4}, {1, 3, 5, 5}, {1, 3, 6, 6},
                                                              {1, 3, 7, 7}, {1, 3, 5, 5}, {1, 3, 4, 4}}}),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(false),
                                 ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref_any"}),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(ov::AnyMap{},
                                                   ov::AnyMap{ov::hint::inference_precision(ov::element::f32)})),
                         AvgPoolingV14LayerCPUTest::getTestCaseName);

const std::vector<poolSpecificParams> paramsAvgV14CeilTorchDynamicAcl = {
        {utils::PoolingTypes::AVG, {2, 2}, {2, 2}, {0, 0}, {0, 0},
         ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::EXPLICIT, false},
        {utils::PoolingTypes::AVG, {2, 2}, {2, 2}, {0, 0}, {0, 0},
         ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::EXPLICIT, true},
        {utils::PoolingTypes::AVG, {3, 3}, {2, 2}, {1, 1}, {1, 1},
         ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::EXPLICIT, true},
        {utils::PoolingTypes::AVG, {3, 3}, {1, 1}, {0, 0}, {0, 0},
         ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::EXPLICIT, true},
};

INSTANTIATE_TEST_SUITE_P(smoke_AvgPoolV14CeilTorchDynamicAcl_CPU_4D, AvgPoolingV14LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsAvgV14CeilTorchDynamicAcl),
                                 ::testing::Values(InputShape{{1, 3, -1, -1},
                                                             {{1, 3, 4, 4}, {1, 3, 5, 5}, {1, 3, 5, 6},
                                                              {1, 3, 6, 5}, {1, 3, 7, 7}, {1, 3, 4, 4}}}),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(false),
                                 ::testing::Values(CPUSpecificParams{{}, {}, {}, "acl"}),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(ov::AnyMap{},
                                                   ov::AnyMap{ov::hint::inference_precision(ov::element::f32)})),
                         AvgPoolingV14LayerCPUTest::getTestCaseName);

// oneDNN's ACL backend handles this padded include-padding case accurately.
INSTANTIATE_TEST_SUITE_P(smoke_AvgPoolV14CeilTorchDynamicPadding_CPU_4D, AvgPoolingV14LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Values(poolSpecificParams{
                                         utils::PoolingTypes::AVG, {3, 3}, {2, 2}, {1, 1}, {1, 1},
                                         ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::EXPLICIT, false}),
                                 ::testing::Values(InputShape{{1, 3, -1, -1},
                                                             {{1, 3, 4, 4}, {1, 3, 5, 5}, {1, 3, 5, 6},
                                                              {1, 3, 6, 5}, {1, 3, 7, 7}, {1, 3, 4, 4}}}),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(false),
                                 ::testing::Values(CPUSpecificParams{{}, {}, {}, "acl"}),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(ov::AnyMap{},
                                                   ov::AnyMap{ov::hint::inference_precision(ov::element::f32)})),
                         AvgPoolingV14LayerCPUTest::getTestCaseName);

const std::vector<maxPoolV8SpecificParams>& paramsMaxV144D_2x2kernel = {
        maxPoolV8SpecificParams{ {2, 2}, {2, 2}, {1, 1}, {0, 0}, {0, 0},
                                                          ov::element::Type_t::i32, 0,
                                                          ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::SAME_UPPER },
        maxPoolV8SpecificParams{ {2, 2}, {2, 2}, {1, 1}, {0, 0}, {0, 0},
                                                          ov::element::Type_t::i32, 0,
                                                          ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::SAME_LOWER }
};

INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolV14_CPU_4D_2x2Kernel, MaxPoolingV14LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsMaxV144D_2x2kernel),
                                 ::testing::ValuesIn(inputShapes4D()),
                                 ::testing::ValuesIn((inpOutPrecision())),
                                 ::testing::ValuesIn(filterCPUInfo(vecCpuConfigsFusing_4D())),
                                 ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MaxPoolingV14LayerCPUTest::getTestCaseName);

const std::vector<maxPoolV8SpecificParams>& paramsMaxV144D_non2x2kernel = {
            maxPoolV8SpecificParams{ {11, 7}, {2, 2}, {1, 1}, {2, 2}, {2, 2},
                                                            ov::element::Type_t::i32, 0,
                                                            ov::op::RoundingType::CEIL_TORCH, ov::op::PadType::EXPLICIT},
};

//The test checks that fallback to nGraph works for ACL non-supported cases
INSTANTIATE_TEST_SUITE_P(smoke_MaxPoolV14_CPU_4D_non2x2Kernel_ref, MaxPoolingV14LayerCPUTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(paramsMaxV144D_non2x2kernel),
                                 ::testing::ValuesIn(inputShapes4D()),
                                 ::testing::ValuesIn((inpOutPrecision())),
                                 ::testing::Values(CPUSpecificParams{{}, {}, {"ref_any"}, "ref_any"}),
                                 ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MaxPoolingV14LayerCPUTest::getTestCaseName);

}  // namespace Pooling
}  // namespace test
}  // namespace ov
