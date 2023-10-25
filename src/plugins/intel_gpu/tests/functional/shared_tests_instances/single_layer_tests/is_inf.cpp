// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/is_inf.hpp"

namespace {
using ov::test::IsInfLayerTest;

const std::vector<std::vector<ov::test::InputShape>> inShapesStatic = {
    {{{}, {{2}}}},
    {{{}, {{10, 200}}}},
    {{{}, {{4, 4, 16}}}},
    {{{}, {{2, 17, 5, 4}}}},
    {{{}, {{16, 16, 16, 16, 16}}}},
    {{{}, {{16, 16, 16, 16, 16, 16}}}},
};

constexpr std::array<ov::element::Type, 2> netPrecisions = {ov::element::f32, ov::element::f16};

constexpr std::array<bool, 2> detectNegative = {true, false};

constexpr std::array<bool, 2> detectPositive = {true, false};

const std::map<std::string, std::string> additional_config = {};

const auto isInfParams = ::testing::Combine(::testing::ValuesIn(inShapesStatic),
                                            ::testing::ValuesIn(detectNegative),
                                            ::testing::ValuesIn(detectPositive),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values(additional_config));

TEST_P(IsInfLayerTest, CompareWithRefs) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_static, IsInfLayerTest, isInfParams, IsInfLayerTest::getTestCaseName);

}  // namespace
