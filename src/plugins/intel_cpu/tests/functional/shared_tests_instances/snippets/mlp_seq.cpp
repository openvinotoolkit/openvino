// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mlp.hpp"

#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

std::vector<std::vector<InputShape>> transposedShape_2D() {
    auto shapes = SNIPPETS_TESTS_STATIC_SHAPES(
        {{64, 64}});
    return shapes;
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MLP_SEQ_2D,
                         MLP,
                         ::testing::Combine(::testing::ValuesIn(transposedShape_2D()),
                                            ::testing::ValuesIn(precision_f32(1)),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(MLP::default_thread_count),
                                            ::testing::Values(1),  // Subgraph
                                            ::testing::Values(1),  // MLP
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MLP::getTestCaseName);
}  // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
