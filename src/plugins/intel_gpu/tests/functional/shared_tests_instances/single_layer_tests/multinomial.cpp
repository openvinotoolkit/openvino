// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "single_layer_tests/multinomial.hpp"

using namespace ov::test::subgraph;

namespace {

const std::vector<ov::test::ElementType> netPrecisions = {
    ov::element::f16,
    ov::element::f32,
};

const std::vector<ov::Shape> inputShapes = {
    {1, 32},
    {2, 28},
};

const std::vector<int64_t> numSamples = {
    2, 4,
};

const std::vector<bool> withReplacement = {
    false,
    true
};

const std::vector<bool> logProbes = {
    false,
    true
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Multinomial,
    MultinomialTest,
    testing::Combine(testing::ValuesIn(netPrecisions),
                     ::testing::Values(ov::element::undefined),
                     ::testing::Values(ov::element::undefined),
                     testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes)),
                     testing::ValuesIn(numSamples),
                     testing::Values(ov::element::i64),
                     testing::ValuesIn(withReplacement),
                     testing::ValuesIn(logProbes),
                     testing::Values(ov::test::utils::DEVICE_GPU),
                     testing::Values(ov::AnyMap())),
                     MultinomialTest::getTestCaseName);
} // anonymous namespace
