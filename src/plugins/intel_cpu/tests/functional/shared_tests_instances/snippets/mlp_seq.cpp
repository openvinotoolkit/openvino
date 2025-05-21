// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mlp.hpp"

#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

std::vector<std::vector<InputShape>> inputShape_2D(bool with_dynamic = true) {
    auto shapes = SNIPPETS_TESTS_STATIC_SHAPES(
        {{1, 64}},
        {{2, 64}},
        {{4, 64}},
        {{8, 64}});
    if (with_dynamic) {
        shapes.push_back({{PartialShape{-1, 64}, {{1, 64}, {8, 64}, {8, 64}, {6, 64}}}});
        shapes.push_back({{PartialShape{-1, 64}, {{2, 64}, {2, 64}, {4, 64}, {3, 64}}}});
    }
    return shapes;
}

// Returns a vector of pairs where:
//   - The first element is the number of hidden layers in the MLP
//   - The second element is a pair: {expected number of subgraphs, expected number of nodes}
std::vector<std::pair<size_t, std::pair<size_t, size_t>>> numHiddenLayersWithExpectations() {
    return {
        {1, {1, 1}},
        {3, {2, 2}},
        {5, {3, 3}},
    };
}

std::vector<std::pair<size_t, std::pair<size_t, size_t>>> numHiddenLayersWithExpectationsBf16() {
    return {
        {1, {3, 9}},
        {3, {3, 13}},
        {5, {5, 18}},
    };
}

std::vector<std::pair<size_t, std::pair<size_t, size_t>>> numHiddenLayersWithExpectationsQuantized() {
    return {
        {1, {1, 1}},
        {3, {1, 1}},
        {5, {1, 1}},
    };
}

std::vector<size_t> hiddenMatmulSizes() {
    return {64, 128, 256};
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MLP_SEQ_2D_f32,
                         MLP,
                         ::testing::Combine(::testing::ValuesIn(inputShape_2D()),
                                            ::testing::ValuesIn(precision_f32(1)),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(MLP::default_thread_count),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config),
                                            ::testing::ValuesIn(numHiddenLayersWithExpectations()),
                                            ::testing::ValuesIn(hiddenMatmulSizes())),
                         MLP::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MLP_SEQ_2D_f32_prc_bf16,
                         MLP,
                         ::testing::Combine(::testing::ValuesIn(inputShape_2D()),
                                            ::testing::ValuesIn(precision_f32(1)),
                                            ::testing::Values(ov::element::bf16),
                                            ::testing::Values(MLP::default_thread_count),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config),
                                            ::testing::ValuesIn(numHiddenLayersWithExpectationsBf16()),
                                            ::testing::ValuesIn(hiddenMatmulSizes())),
                         MLP::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MLP_SEQ_Quantized_2D_f32,
                         MLPQuantized,
                         ::testing::Combine(::testing::ValuesIn(inputShape_2D()),
                                            ::testing::ValuesIn(precision_f32(1)),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(MLPQuantized::default_thread_count),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config),
                                            ::testing::ValuesIn(numHiddenLayersWithExpectationsQuantized()),
                                            ::testing::ValuesIn(hiddenMatmulSizes())),
                         MLPQuantized::getTestCaseName);

}  // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
