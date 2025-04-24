// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/squeeze_unsqueeze.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::SqueezeUnsqueezeLayerTest;
using ov::test::utils::SqueezeOpType;

std::map<std::vector<ov::Shape>, std::vector<std::vector<int>>> axesVectors = {
        {{{1, 1, 1, 1}}, {{-1}, {0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {0, 3}, {1, 2}, {2, 3}, {0, 1, 2}, {0, 2, 3}, {1, 2, 3}, {0, 1, 2, 3}}},
        {{{1, 2, 3, 4}}, {{0}}},
        {{{2, 1, 3, 4}}, {{1}}},
        {{{1}}, {{-1}, {0}}},
        {{{1, 2}}, {{0}}},
        {{{2, 1}}, {{1}, {-1}}},
};

std::map<std::vector<ov::Shape>, std::vector<std::vector<int>>> emptyAxesVectors = {
        {{{1, 1, 1, 1}}, {{}}},
        {{{1, 2, 3, 4}}, {{}}},
        {{{2, 1, 3, 4}}, {{}}},
        {{{1}}, {{}}},
        {{{1, 2}}, {{}}},
        {{{2, 1}}, {{}}},
};

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16
};

const std::vector<SqueezeOpType> opTypes = {
        SqueezeOpType::SQUEEZE,
        SqueezeOpType::UNSQUEEZE
};

auto prepare_cases = [](const std::vector<std::pair<std::vector<ov::Shape>, std::vector<int>>>& raw_axes) {
        std::vector<std::pair<std::vector<ov::test::InputShape>, std::vector<int>>> cases;
        for (const auto& raw_case : raw_axes)
                cases.emplace_back(ov::test::static_shapes_to_test_representation(raw_case.first),
                                   raw_case.second);
        return cases;
};

INSTANTIATE_TEST_SUITE_P(smoke_Basic, SqueezeUnsqueezeLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(prepare_cases(ov::test::utils::combineParams(axesVectors))),
                                ::testing::ValuesIn(opTypes),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        SqueezeUnsqueezeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Basic_emptyAxes, SqueezeUnsqueezeLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(prepare_cases(ov::test::utils::combineParams(emptyAxesVectors))),
                                ::testing::Values(SqueezeOpType::SQUEEZE),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        SqueezeUnsqueezeLayerTest::getTestCaseName);
}  // namespace
