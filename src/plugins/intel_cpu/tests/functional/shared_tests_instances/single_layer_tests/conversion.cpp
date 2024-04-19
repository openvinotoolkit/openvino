// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_op_tests/conversion.hpp"

namespace {
using ov::test::ConversionLayerTest;

const std::vector<ov::test::utils::ConversionTypes> conversionOpTypes = {
    ov::test::utils::ConversionTypes::CONVERT,
    ov::test::utils::ConversionTypes::CONVERT_LIKE,
};

const std::vector<std::vector<ov::Shape>> shapes = {{{1, 2, 3, 4}}};

const std::vector<ov::element::Type> types = {
    ov::element::u8,
    ov::element::i8,
    ov::element::u16,
    ov::element::i16,
    ov::element::u32,
    ov::element::i32,
    ov::element::u64,
    ov::element::i64,
    ov::element::bf16,
    ov::element::f16,
    ov::element::f32,
    ov::element::f64,
};

INSTANTIATE_TEST_SUITE_P(smoke_ConversionLayerTest,
                         ConversionLayerTest,
                         ::testing::Combine(::testing::ValuesIn(conversionOpTypes),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes)),
                                            ::testing::ValuesIn(types),
                                            ::testing::ValuesIn(types),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConversionToBooleanLayerTest,
                         ConversionLayerTest,
                         ::testing::Combine(::testing::ValuesIn(conversionOpTypes),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes)),
                                            ::testing::ValuesIn(types),
                                            ::testing::Values(ov::element::boolean),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ConversionLayerTest::getTestCaseName);
}  // namespace
