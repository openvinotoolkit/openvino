// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/is_inf.hpp"

#include <array>
#include <vector>

using namespace ov::test;
using namespace ov::test::subgraph;

namespace {

const std::vector<std::vector<InputShape>> inShapesStatic = {
    {{{}, {{2}}}},
    {{{}, {{10, 200}}}},
    {{{}, {{4, 4, 16}}}},
    {{{}, {{2, 17, 5, 4}}}},
    {{{}, {{16, 16, 16, 16, 16}}}},
    {{{}, {{16, 16, 16, 16, 16, 16}}}},
};

constexpr std::array<ElementType, 2> netPrecisions = {ov::element::f32, ov::element::f16};

constexpr std::array<bool, 2> detectNegative = {true, false};

constexpr std::array<bool, 2> detectPositive = {true, false};

const std::map<std::string, std::string> additional_config = {};

const auto isInfParams = ::testing::Combine(::testing::ValuesIn(inShapesStatic),
                                            ::testing::ValuesIn(detectNegative),
                                            ::testing::ValuesIn(detectPositive),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                            ::testing::Values(additional_config));

TEST_P(IsInfLayerTest, CompareWithRefs) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_static, IsInfLayerTest, isInfParams, IsInfLayerTest::getTestCaseName);

}  // namespace
