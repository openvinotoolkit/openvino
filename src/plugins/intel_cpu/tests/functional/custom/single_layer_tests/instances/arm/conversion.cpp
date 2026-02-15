// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/conversion.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Conversion {

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_7D_Dynamic, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_7D_dynamic()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::Values(ov::test::SpecialValue::none),
                                ::testing::Values(CPUSpecificParams({}, {}, {}, {}))),
                        ConvertCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_7D_Static, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_7D_static()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::Values(ov::test::SpecialValue::none),
                                ::testing::Values(CPUSpecificParams({}, {}, {}, {}))),
                        ConvertCPULayerTest::getTestCaseName);

}  // namespace Conversion
}  // namespace test
}  // namespace ov
