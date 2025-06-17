// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mlp_seq.hpp"

#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

std::vector<std::vector<InputShape>> inputShape_2D() {
    auto shapes = SNIPPETS_TESTS_STATIC_SHAPES(
        {{1, 64}},
        {{2, 64}},
        {{4, 64}},
        {{8, 64}});
    shapes.push_back({{PartialShape{-1, 64}, {{1, 64}, {8, 64}, {6, 64}, {8, 64}}}});
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
        {1, {3, 3}}, // In Convert + MLP + Out Convert
        {3, {3, 3}}, // In Convert + MLP + Out Convert
        {5, {3, 3}}, // In Convert + MLP + Out Convert
        {7, {4, 4}}, // In Convert + MLP_1 + MLP_2 + Out Convert
    };
}

std::vector<std::pair<size_t, std::pair<size_t, size_t>>> numHiddenLayersWithExpectationsQuantized() {
    return {
        {1, {1, 1}},
        {3, {1, 1}},
        {5, {1, 1}},
        {7, {2, 2}},
    };
}

std::vector<size_t> hiddenMatmulSizes() {
    return {64, 128, 256};
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MLP_SEQ_2D_f32,
                         MLPSeq,
                         ::testing::Combine(::testing::ValuesIn(inputShape_2D()),
                                            ::testing::ValuesIn(precision_f32(1)),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(MLPSeq::default_thread_count),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config),
                                            ::testing::ValuesIn(numHiddenLayersWithExpectations()),
                                            ::testing::ValuesIn(hiddenMatmulSizes())),
                         MLPSeq::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MLP_SEQ_2D_f32_prc_bf16,
                         MLPSeq,
                         ::testing::Combine(::testing::ValuesIn(inputShape_2D()),
                                            ::testing::ValuesIn(precision_f32(1)),
                                            ::testing::Values(ov::element::bf16),
                                            ::testing::Values(MLPSeq::default_thread_count),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config),
                                            ::testing::ValuesIn(numHiddenLayersWithExpectationsBf16()),
                                            ::testing::ValuesIn(hiddenMatmulSizes())),
                         MLPSeq::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MLP_SEQ_Quantized_2D_f32,
                         MLPSeqQuantized,
                         ::testing::Combine(::testing::ValuesIn(inputShape_2D()),
                                            ::testing::ValuesIn(precision_f32(1)),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(MLPSeqQuantized::default_thread_count),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config),
                                            ::testing::ValuesIn(numHiddenLayersWithExpectationsQuantized()),
                                            ::testing::ValuesIn(hiddenMatmulSizes())),
                         MLPSeqQuantized::getTestCaseName);

}  // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
