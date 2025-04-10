// Copyright (C) 2018-2025 Intel Corporation
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

const std::vector<ov::element::Type> floatTypes = {
    ov::element::f32,
    ov::element::f16,
    ov::element::bf16,
};

const std::vector<ov::element::Type> f8Types = {
    ov::element::f8e4m3,
    ov::element::f8e5m2,
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

INSTANTIATE_TEST_SUITE_P(smoke_ConversionToF8LayerTest,
                         ConversionLayerTest,
                         ::testing::Combine(::testing::Values(conversionOpTypes[0]),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes)),
                                            ::testing::ValuesIn(floatTypes),
                                            ::testing::ValuesIn(f8Types),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConversionFromF8LayerTest,
                         ConversionLayerTest,
                         ::testing::Combine(::testing::Values(conversionOpTypes[0]),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(shapes)),
                                            ::testing::ValuesIn(f8Types),
                                            ::testing::ValuesIn(floatTypes),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ConversionLayerTest::getTestCaseName);

}  // namespace
