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
namespace {

INSTANTIATE_TEST_SUITE_P(smoke_Col2ImLayoutTestBF16, Col2ImLayerCPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(col2ImParamsVector),
                        ::testing::Values(ElementType::bf16),
                        ::testing::ValuesIn(indexPrecisions),
                        ::testing::Values(ov::test::utils::DEVICE_CPU)),
                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref_bf16"})),
                Col2ImLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace Col2Im
}  // namespace test
}  // namespace ov
