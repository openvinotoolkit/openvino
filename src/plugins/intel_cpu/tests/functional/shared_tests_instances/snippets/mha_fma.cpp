// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mha.hpp"

#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHAMulAdd,
    MHAMulAdd,
    ::testing::Combine(
        ::testing::ValuesIn(SNIPPETS_TESTS_STATIC_SHAPES({{1, 10, 12, 16}, {1, 10, 12, 16}, {1, 10, 12, 16}})),
        ::testing::ValuesIn(precision_f32(3)),
        ::testing::Values(ov::element::f32),
        ::testing::ValuesIn({false}),  // Need to support True for graph builder in tests
        ::testing::Values(MHA::default_thread_count),
        ::testing::Values(1),
        ::testing::Values(1),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

}  // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
