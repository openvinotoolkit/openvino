// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mlp.hpp"

#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

std::vector<std::vector<InputShape>> inputShape_2D() {
    auto shapes = SNIPPETS_TESTS_STATIC_SHAPES(
        {{64, 64}});
    return shapes;
}

std::vector <size_t> numHiddenLayers() {
    return {2, 3, 5, 7, 9};
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MLP_SEQ_2D_f32,
                         MLP,
                         ::testing::Combine(::testing::ValuesIn(inputShape_2D()),
                                            ::testing::ValuesIn(precision_f32(1)),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn({static_cast<size_t>(1), MLP::default_thread_count}),
                                            ::testing::Values(1),  // Subgraph
                                            ::testing::Values(1),  // MLP
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config),
                                            ::testing::Values(2),
                                            ::testing::ValuesIn(numHiddenLayers())),
                         MLP::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MLP_SEQ_2D_i8,
                         MLP,
                         ::testing::Combine(::testing::ValuesIn(inputShape_2D()),
                                            ::testing::ValuesIn(precision_i8(1)),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::ValuesIn({static_cast<size_t>(1), MLP::default_thread_count}),
                                            ::testing::Values(2),  // Subgraph
                                            ::testing::Values(2),  // MLP
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config),
                                            ::testing::Values(2),
                                            ::testing::ValuesIn(numHiddenLayers())),
                         MLP::getTestCaseName);
}  // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
