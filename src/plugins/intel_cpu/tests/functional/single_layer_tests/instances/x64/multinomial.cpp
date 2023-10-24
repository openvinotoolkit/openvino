// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/multinomial.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions{
namespace Multinomial{

const std::vector<ov::test::ElementType> convert_types = {
    ov::test::ElementType::i32,
    ov::test::ElementType::i64,
};

const std::vector<bool> with_replacements = {true, false};

const std::vector<bool> log_probs = {true, false};

const std::vector<InputShape> probs_static = {
    {{4, 4}, {{4, 4}}},
    {{2, 7}, {{2, 7}}},
};

const std::vector<InputShape> probs_dynamic = {
    {{-1, -1}, {{4, 4}}},
    {{-1, -1}, {{2, 7}}},
};

const std::vector<InputShape> num_samples_static = {
    {{1}, {{1}}},
    {{1}, {{1}}},
};

const std::vector<InputShape> num_samples_dynamic = {
    {{-1}, {{1}}},
    {{-1}, {{1}}},
};

const auto params_static = ::testing::Combine(::testing::ValuesIn(probs_static),
                                              ::testing::ValuesIn(num_samples_static),
                                              ::testing::ValuesIn(convert_types),
                                              ::testing::ValuesIn(with_replacements),
                                              ::testing::ValuesIn(log_probs),
                                              ::testing::Values(1),
                                              ::testing::Values(1),
                                              ::testing::Values(emptyCPUSpec),
                                              ::testing::Values(empty_plugin_config));

const auto params_dynamic = ::testing::Combine(::testing::ValuesIn(probs_dynamic),
                                               ::testing::ValuesIn(num_samples_dynamic),
                                               ::testing::ValuesIn(convert_types),
                                               ::testing::ValuesIn(with_replacements),
                                               ::testing::ValuesIn(log_probs),
                                               ::testing::Values(1),
                                               ::testing::Values(1),
                                               ::testing::Values(emptyCPUSpec),
                                               ::testing::Values(empty_plugin_config));

INSTANTIATE_TEST_SUITE_P(smoke_MultinomialStatic,
                         MultinomialLayerTestCPU,
                         params_static,
                         MultinomialLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MultinomialDynamic,
                         MultinomialLayerTestCPU,
                         params_dynamic,
                         MultinomialLayerTestCPU::getTestCaseName);

} // namespace Multinomial
} // namespace CPULayerTestsDefinitions