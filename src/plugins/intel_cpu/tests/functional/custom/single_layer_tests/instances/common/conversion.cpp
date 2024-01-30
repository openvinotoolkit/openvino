// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/conversion.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Conversion {

static std::string expectedPrimitiveType() {
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
    return "ref";
#endif
#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
    return "acl";
#endif
    return {};
}

std::vector<CPUSpecificParams> memForm4D_dynamic = {
    CPUSpecificParams({nchw}, {nchw}, {}, expectedPrimitiveType()),
    CPUSpecificParams({nhwc}, {nhwc}, {}, expectedPrimitiveType()),
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_4D_Dynamic, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_dynamic()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::ValuesIn(memForm4D_dynamic),
                                ::testing::Values(false)),
                        ConvertCPULayerTest::getTestCaseName);

std::vector<CPUSpecificParams> memForm4D_static_common = {
    CPUSpecificParams({nchw}, {nchw}, {}, {}),
    CPUSpecificParams({nhwc}, {nhwc}, {}, {}),
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_4D_Static, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_static()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::ValuesIn(precisions()),
                                ::testing::ValuesIn(memForm4D_static_common),
                                ::testing::Values(false)),
                        ConvertCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_clampTestCase,
                         ConvertCPULayerTest,
                         ::testing::Combine(::testing::Values(InputShape({{1, 2, 3, 4}, {{1, 2, 3, 4}}})),
                                            ::testing::ValuesIn(precisions()),
                                            ::testing::ValuesIn(precisions()),
                                            ::testing::Values(CPUSpecificParams({}, {}, {}, {})),
                                            ::testing::Values(true)),
                         ConvertCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_clampTestCase_f16,
                         ConvertCPULayerTest,
                         ::testing::Combine(::testing::Values(InputShape({{1, 2, 3, 4}, {{1, 2, 3, 4}}})),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(CPUSpecificParams({}, {}, {}, {})),
                                            ::testing::Values(true)),
                         ConvertCPULayerTest::getTestCaseName);
}  // namespace Conversion
}  // namespace test
}  // namespace ov
