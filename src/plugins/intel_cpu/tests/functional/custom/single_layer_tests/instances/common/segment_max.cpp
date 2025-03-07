// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/segment_max.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace SegmentMax {
INSTANTIATE_TEST_SUITE_P(smoke_SegmentMaxLayoutTestF32, SegmentMaxLayerCPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(SegmentMaxParamsVector),
                        ::testing::ValuesIn(std::vector<ElementType>{ElementType::f32, ElementType::f16}),
                        ::testing::Bool(),
                        ::testing::ValuesIn(secondaryInputTypes),
                        ::testing::Values(ov::test::utils::DEVICE_CPU)),
                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref_f32"})),
                SegmentMaxLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SegmentMaxLayoutTestI32, SegmentMaxLayerCPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(SegmentMaxParamsVector),
                        ::testing::Values(ElementType::i32),
                        ::testing::Bool(),
                        ::testing::ValuesIn(secondaryInputTypes),
                        ::testing::Values(ov::test::utils::DEVICE_CPU)),
                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref_i32"})),
                SegmentMaxLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SegmentMaxLayoutTestUI8, SegmentMaxLayerCPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(SegmentMaxParamsVector),
                        ::testing::Values(ElementType::u8, ElementType::i8),
                        ::testing::Bool(),
                        ::testing::ValuesIn(secondaryInputTypes),
                        ::testing::Values(ov::test::utils::DEVICE_CPU)),
                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref_i8"})),
                SegmentMaxLayerCPUTest::getTestCaseName);
}  // namespace SegmentMax
}  // namespace test
}  // namespace ov
