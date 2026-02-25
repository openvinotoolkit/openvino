// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/identity.hpp"

#include "common_test_utils/test_constants.hpp"
#include "shared_test_classes/single_op/identity.hpp"

namespace {
using ov::test::IdentityLayerTest;

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i64,
        ov::element::i32,
        ov::element::i16,
        ov::element::i8,
        ov::element::u8,
};

/**
 * 4D Identity tests
 */
const std::vector<std::vector<ov::Shape>> inputShapes = {
        {{1, 3, 100, 100}},
        {{2, 8, 64, 64}},
        {{2, 5, 64, 64}},
        {{2, 8, 64, 5}},
        {{2, 5, 64, 5}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Identity,
                         IdentityLayerTest,
                         testing::Combine(testing::ValuesIn(netPrecisions),
                                          testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes)),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         IdentityLayerTest::getTestCaseName);

/**
 * 5D Identity tests
 */
const std::vector<std::vector<ov::Shape>> inputShapes5D = {
        {{2, 3, 4, 12, 64}},
        {{2, 5, 11, 32, 32}},
        {{2, 8, 64, 32, 5}},
        {{2, 5, 64, 32, 5}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Identity_5D,
                         IdentityLayerTest,
                         testing::Combine(testing::ValuesIn(netPrecisions),
                                          testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes5D)),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         IdentityLayerTest::getTestCaseName);

}  // namespace
