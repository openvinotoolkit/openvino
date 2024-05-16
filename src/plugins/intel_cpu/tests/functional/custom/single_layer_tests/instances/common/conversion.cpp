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
                                ::testing::ValuesIn(memForm4D_dynamic)),
                        ConvertCPULayerTest::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_4bit_Dynamic, ConvertCPULayerTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes_4D_dynamic()),
                                            ::testing::ValuesIn({ov::element::u4, ov::element::i4}),
                                            ::testing::ValuesIn({ov::element::f32, ov::element::bf16, ov::element::u8, ov::element::i8}),
                                            ::testing::Values(CPUSpecificParams({nchw}, {nchw}, {}, {"ref"}))),
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
                                ::testing::ValuesIn(memForm4D_static_common)),
                        ConvertCPULayerTest::getTestCaseName);

}  // namespace Conversion
}  // namespace test
}  // namespace ov