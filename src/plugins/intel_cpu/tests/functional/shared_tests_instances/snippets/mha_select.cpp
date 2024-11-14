// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mha.hpp"

#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

const auto& inputShapeSelect = SNIPPETS_TESTS_STATIC_SHAPES(
    // without broadcast
    {{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 12, 128, 128}, {1, 12, 128, 128}, {1, 128, 12, 64}},
    {{1, 94, 12, 54}, {1, 94, 12, 54}, {1, 12, 94, 94}, {1, 12, 94, 94}, {1, 12, 94, 94}, {1, 94, 12, 54}},
    // with broadcast
    {{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 12, 1, 1}, {1, 12, 1, 1}, {1, 128, 12, 64}},
    {{2, 52, 6, 102}, {2, 52, 6, 102}, {1, 6, 52, 52}, {1, 6, 1, 1}, {1, 6, 1, 1}, {2, 52, 6, 102}}
);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHA,
    MHASelect,
    ::testing::Combine(::testing::ValuesIn(inputShapeSelect),
                       ::testing::ValuesIn(precision_f32(6)),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(false),  // Need to support True for graph builder in tests
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(2),  // Less + MHA
                       ::testing::Values(2),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

}  // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
