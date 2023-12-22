// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/conversion.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ConversionLayerTest;
const std::vector<ov::test::utils::ConversionTypes> conversionOpTypes = {
    ov::test::utils::ConversionTypes::CONVERT,
    ov::test::utils::ConversionTypes::CONVERT_LIKE,
};

const std::vector<std::vector<ov::Shape>> inShape = {{{1, 2, 3, 4}}};

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16,
        ov::element::u8,
        ov::element::i8,
};

INSTANTIATE_TEST_SUITE_P(smoke_NoReshape, ConversionLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(conversionOpTypes),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShape)),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ConversionLayerTest::getTestCaseName);

}  // namespace
