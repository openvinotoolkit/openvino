// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/eltwise.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Eltwise {
namespace {

const std::vector<std::vector<InputShape>> bitwise_in_shapes_4D = {
    {{{1, -1, -1, -1}, {{1, 3, 2, 2}, {1, 3, 1, 1}}}, {{1, 3, 2, 2}, {{1, 3, 2, 2}}}},
    {{{1, -1, -1, -1}, {{1, 64, 2, 2}, {1, 64, 1, 1}}}, {{1, 64, 2, 2}, {{1, 64, 2, 2}}}},
};

const std::vector<CPUSpecificParams>& bitwiseSnippetsCpuParams() {
    static const std::vector<CPUSpecificParams> params = {CPUSpecificParams({}, {}, {}, "jit")};
    return params;
}

const auto params_4D_bitwise_snippets =
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(bitwise_in_shapes_4D),
                                          ::testing::ValuesIn({utils::EltwiseTypes::BITWISE_AND,
                                                               utils::EltwiseTypes::BITWISE_OR,
                                                               utils::EltwiseTypes::BITWISE_XOR}),
                                          ::testing::ValuesIn(secondaryInputTypes()),
                                          ::testing::ValuesIn({utils::OpType::VECTOR}),
                                          ::testing::ValuesIn({ElementType::i8, ElementType::u8}),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU),
                                          ::testing::Values(ov::AnyMap())),
                       ::testing::ValuesIn(bitwiseSnippetsCpuParams()),
                       ::testing::Values(emptyFusingSpec),
                       ::testing::Values(true));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Bitwise_Snippets,
                         EltwiseLayerCPUTest,
                         params_4D_bitwise_snippets,
                         EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_bitwise_scalar_snippets =
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(bitwise_in_shapes_4D),
                                          ::testing::ValuesIn({utils::EltwiseTypes::BITWISE_AND,
                                                               utils::EltwiseTypes::BITWISE_OR,
                                                               utils::EltwiseTypes::BITWISE_XOR}),
                                          ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                          ::testing::Values(utils::OpType::SCALAR),
                                          ::testing::ValuesIn({ElementType::i8, ElementType::u8}),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU),
                                          ::testing::Values(ov::AnyMap())),
                       ::testing::ValuesIn(bitwiseSnippetsCpuParams()),
                       ::testing::Values(emptyFusingSpec),
                       ::testing::Values(true));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Bitwise_Scalar_Snippets,
                         EltwiseLayerCPUTest,
                         params_4D_bitwise_scalar_snippets,
                         EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_bitwise_NOT_snippets =
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(bitwise_in_shapes_4D),
                                          ::testing::ValuesIn({utils::EltwiseTypes::BITWISE_NOT}),
                                          ::testing::ValuesIn({ov::test::utils::InputLayerType::CONSTANT}),
                                          ::testing::ValuesIn({utils::OpType::VECTOR}),
                                          ::testing::ValuesIn({ElementType::i8, ElementType::u8}),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU),
                                          ::testing::Values(ov::AnyMap())),
                       ::testing::ValuesIn(bitwiseSnippetsCpuParams()),
                       ::testing::Values(emptyFusingSpec),
                       ::testing::Values(true));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Bitwise_NOT_Snippets,
                         EltwiseLayerCPUTest,
                         params_4D_bitwise_NOT_snippets,
                         EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_int_jit = ::testing::Combine(
    ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D())),
        ::testing::ValuesIn({utils::EltwiseTypes::ADD, utils::EltwiseTypes::MULTIPLY}),
        ::testing::ValuesIn(secondaryInputTypes()),
        ::testing::ValuesIn(opTypes()),
        ::testing::ValuesIn({ElementType::i8, ElementType::u8, ElementType::f16, ElementType::i32, ElementType::f32}),
        ::testing::Values(ov::element::dynamic),
        ::testing::Values(ov::element::dynamic),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(additional_config())),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_int_jit, EltwiseLayerCPUTest, params_4D_int_jit, EltwiseLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace Eltwise
}  // namespace test
}  // namespace ov
