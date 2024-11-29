// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mha.hpp"

#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

const auto& inputShapesExtractedReshape = SNIPPETS_TESTS_STATIC_SHAPES(
    {{2, 196, 64}, {2, 64, 196}, {2, 14, 14, 14, 1}, {2, 14, 14, 1, 14}, {2, 196, 64}},
    {{1, 16, 10}, {1, 10, 16}, {1, 4, 4, 4, 1}, {1, 4, 4, 1, 4}, {1, 16, 10}},
    {{1, 16, 10}, {1, 10, 16}, {1, 1, 1, 1, 1}, {1, 4, 4, 4, 4}, {1, 16, 10}},
    {{1, 16, 10}, {1, 10, 16}, {1, 4, 4, 4, 4}, {1, 1, 1, 1, 1}, {1, 16, 10}},
    {{1, 4, 16, 10}, {1, 4, 10, 16}, {1, 4, 256}, {1, 4, 256}, {1, 4, 16, 10}},
    {{1, 4, 16, 10}, {1, 4, 10, 16}, {1, 1, 256}, {1, 4, 1}, {1, 4, 16, 10}});

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAWithExtractedReshape,
    MHAWithExtractedReshape,
    ::testing::Combine(::testing::ValuesIn(inputShapesExtractedReshape),
                       ::testing::Values(std::vector<element::Type>{}),
                       ::testing::Values(ov::element::f32),
                       ::testing::ValuesIn({true}),  // False is not supported for graph builder in tests
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(3),  // Extracted Add + Extracted Reshape + MHA
                       ::testing::Values(2),  // Extracted Add + MHA
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

}  // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
