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
                                ::testing::Values(ov::test::SpecialValue::none),
                                ::testing::ValuesIn(memForm4D_dynamic)),
                        ConvertCPULayerTest::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_4bit_Dynamic, ConvertCPULayerTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes_4D_dynamic()),
                                            ::testing::ValuesIn({ov::element::u4, ov::element::i4}),
                                            ::testing::ValuesIn({ov::element::f32, ov::element::bf16, ov::element::u8, ov::element::i8}),
                                            ::testing::Values(ov::test::SpecialValue::none),
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
                                ::testing::Values(ov::test::SpecialValue::none),
                                ::testing::ValuesIn(memForm4D_static_common)),
                        ConvertCPULayerTest::getTestCaseName);

const std::vector<ov::element::Type> float_precisions = {
    ov::element::f32,
    ov::element::f16,
    ov::element::bf16,
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_float_to_nf4, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_dynamic()),
                                ::testing::ValuesIn(float_precisions),
                                ::testing::Values(ov::element::nf4),
                                ::testing::Values(ov::test::SpecialValue::none),
                                ::testing::Values(CPUSpecificParams({nchw}, {nchw}, {}, {"ref"}))),
                        ConvertCPULayerTest::getTestCaseName);

const std::vector<ov::element::Type> f8_precisions = {
    ov::element::f8e4m3,
    ov::element::f8e5m2,
};

const std::vector<ov::test::SpecialValue> specialValue = {
    ov::test::SpecialValue::none,
    ov::test::SpecialValue::nan,
    ov::test::SpecialValue::inf,
    ov::test::SpecialValue::overflow,
};

std::vector<CPUSpecificParams> memForm4D_fp8 = {
    CPUSpecificParams({nchw}, {nchw}, {}, expectedPrimitiveType()),
    CPUSpecificParams({nhwc}, {nhwc}, {}, expectedPrimitiveType()),
};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_from_fp8_Static, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_static()),
                                ::testing::ValuesIn(f8_precisions),
                                ::testing::ValuesIn(float_precisions),
                                ::testing::ValuesIn(specialValue),
                                ::testing::ValuesIn(memForm4D_fp8)),
                        ConvertCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_to_fp8_Static, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_static()),
                                ::testing::ValuesIn(float_precisions),
                                ::testing::ValuesIn(f8_precisions),
                                ::testing::ValuesIn(specialValue),
                                ::testing::ValuesIn(memForm4D_fp8)),
                        ConvertCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_from_fp8_Dynamic, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_dynamic()),
                                ::testing::ValuesIn(f8_precisions),
                                ::testing::ValuesIn(float_precisions),
                                ::testing::ValuesIn(specialValue),
                                ::testing::ValuesIn(memForm4D_fp8)),
                        ConvertCPULayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvertCPULayerTest_to_fp8_Dynamic, ConvertCPULayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes_4D_dynamic()),
                                ::testing::ValuesIn(float_precisions),
                                ::testing::ValuesIn(f8_precisions),
                                ::testing::ValuesIn(specialValue),
                                ::testing::ValuesIn(memForm4D_fp8)),
                        ConvertCPULayerTest::getTestCaseName);

}  // namespace Conversion
}  // namespace test
}  // namespace ov