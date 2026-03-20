// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/string_tensor_unpack.hpp"
#include "custom/single_layer_tests/classes/pad_string.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace StringTensorUnpack {
INSTANTIATE_TEST_SUITE_P(smoke_StringTensorUnpackLayoutTest, StringTensorUnpackLayerCPUTest,
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(StringTensorUnpackParamsVector),
                        ::testing::Values(ov::test::utils::DEVICE_CPU)),
                ::testing::Values(CPUSpecificParams{{}, {}, {}, "ref_string"})),
                StringTensorUnpackLayerCPUTest::getTestCaseName);
}  // namespace StringTensorUnpack

namespace PadString {
INSTANTIATE_TEST_SUITE_P(smoke_PadStringLayoutTest, PadStringLayerCPUTest,
              ::testing::Combine(
                  ::testing::Combine(
                      ::testing::ValuesIn(PadStringParamsVector),
                      ::testing::Values(ov::test::utils::DEVICE_CPU)),
                  ::testing::Values(CPUSpecificParams{})),
              PadStringLayerCPUTest::getTestCaseName);
}  // namespace PadString
}  // namespace test
}  // namespace ov
