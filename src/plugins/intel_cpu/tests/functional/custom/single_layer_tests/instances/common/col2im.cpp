// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/col2im.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Col2Im {
INSTANTIATE_TEST_SUITE_P(smoke_Col2ImLayoutTestF32, Col2ImLayerCPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(col2ImParamsVector),
                        ::testing::ValuesIn(std::vector<ElementType>{ElementType::f32, ElementType::f16}),
                        ::testing::ValuesIn(indexPrecisions),
                        ::testing::Values(ov::test::utils::DEVICE_CPU)),
                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref_f32"})),
                Col2ImLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Col2ImLayoutTestI32, Col2ImLayerCPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(col2ImParamsVector),
                        ::testing::Values(ElementType::i32),
                        ::testing::ValuesIn(indexPrecisions),
                        ::testing::Values(ov::test::utils::DEVICE_CPU)),
                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref_i32"})),
                Col2ImLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Col2ImLayoutTestUI8, Col2ImLayerCPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(col2ImParamsVector),
                        ::testing::Values(ElementType::u8, ElementType::i8),
                        ::testing::ValuesIn(indexPrecisions),
                        ::testing::Values(ov::test::utils::DEVICE_CPU)),
                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref_i8"})),
                Col2ImLayerCPUTest::getTestCaseName);
}  // namespace Col2Im
}  // namespace test
}  // namespace ov
