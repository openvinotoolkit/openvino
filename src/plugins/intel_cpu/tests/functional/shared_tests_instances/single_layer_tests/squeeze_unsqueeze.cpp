// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/squeeze_unsqueeze.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::SqueezeUnsqueezeLayerTest;

namespace {
std::map<std::vector<ov::Shape>, std::vector<std::vector<int>>> raw_axes = {
        {{{1, 1, 1, 1}}, {{-1}, {0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {0, 3}, {1, 2}, {2, 3}, {0, 1, 2}, {0, 2, 3}, {1, 2, 3}, {0, 1, 2, 3}}},
        {{{1, 2, 3, 4}}, {{0}}},
        {{{2, 1, 3, 4}}, {{1}}},
        {{{1}}, {{-1}, {0}}},
        {{{1, 2}}, {{0}}},
        {{{2, 1}}, {{1}, {-1}}},
};

std::map<std::vector<ov::Shape>, std::vector<std::vector<int>>> raw_empty_axes = {
        {{{1, 1, 1, 1}}, {{}}},
        {{{1, 2, 3, 4}}, {{}}},
        {{{2, 1, 3, 4}}, {{}}},
        {{{1}}, {{}}},
        {{{1, 2}}, {{}}},
        {{{2, 1}}, {{}}},
};

auto combined_axes = ov::test::utils::combineParams(raw_axes);
auto combined_empty_axes = ov::test::utils::combineParams(raw_empty_axes);

auto prepare_cases = [](const std::vector<std::pair<std::vector<ov::Shape>, std::vector<int>>>& raw_axes) {
        std::vector<std::pair<std::vector<ov::test::InputShape>, std::vector<int>>> cases;
        for (const auto& raw_case : raw_axes)
                cases.emplace_back(ov::test::static_shapes_to_test_representation(raw_case.first),
                                   raw_case.second);
        return cases;
};

auto axes = prepare_cases(combined_axes);
auto empty_axes = prepare_cases(combined_empty_axes);

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16
};

const std::vector<ov::test::utils::SqueezeOpType> opTypes = {
        ov::test::utils::SqueezeOpType::SQUEEZE,
        ov::test::utils::SqueezeOpType::UNSQUEEZE
};

INSTANTIATE_TEST_SUITE_P(smoke_Basic, SqueezeUnsqueezeLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(axes),
                                ::testing::ValuesIn(opTypes),
                                ::testing::ValuesIn(model_types),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        SqueezeUnsqueezeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Basic_emptyAxes, SqueezeUnsqueezeLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(empty_axes),
                                ::testing::Values(ov::test::utils::SqueezeOpType::SQUEEZE),
                                ::testing::ValuesIn(model_types),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        SqueezeUnsqueezeLayerTest::getTestCaseName);
}  // namespace
