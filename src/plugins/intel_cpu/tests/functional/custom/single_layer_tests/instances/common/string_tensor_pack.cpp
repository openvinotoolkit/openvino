// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/string_tensor_pack.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace StringTensorPack {
INSTANTIATE_TEST_SUITE_P(smoke_StringTensorPackLayoutTest, StringTensorPackLayerCPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(StringTensorPackParamsVector),
                        ::testing::ValuesIn(std::vector<ElementType>{ElementType::i32, ElementType::i64}),
                        ::testing::Values(ov::test::utils::DEVICE_CPU)),
                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref_i32"})),
                StringTensorPackLayerCPUTest::getTestCaseName);
}  // namespace StringTensorPack
}  // namespace test
}  // namespace ov
