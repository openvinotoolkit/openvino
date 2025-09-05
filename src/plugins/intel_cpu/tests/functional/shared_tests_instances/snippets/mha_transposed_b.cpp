// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mha.hpp"

#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

std::vector<std::vector<ov::test::InputShape>> inputShapesTransposedB {
    {
        {{}, {{1, 12, 12, 64}}},
        {{}, {{1, 12, 48, 64}}},
        {{}, {{1, 12, 48, 64}}}
    },
    {
        {PartialShape{-1, 3, -1, 64}, {{1, 3, 12, 64}, {2, 3, 36, 64}}},
        {PartialShape{-1, 3, -1, 64}, {{1, 3, 14, 64}, {2, 3, 42, 64}}},
        {PartialShape{-1, 3, -1, -1}, {{1, 3, 14, 36}, {2, 3, 42, 36}}},
    },
    {
        {PartialShape{2, -1, 32, -1}, {{2, 1, 32, 70}, {2, 2, 32, 96}}},
        {PartialShape{2, -1, 49, -1}, {{2, 3, 49, 70}, {2, 1, 49, 96}}},
        {PartialShape{2, -1, 49, -1}, {{2, 1, 49, 17}, {2, 2, 49, 81}}},
    },
};

#ifdef OPENVINO_ARCH_ARM64
static constexpr size_t expected_num_nodes = 2; // MHA + separated Transpose
#else
static constexpr size_t expected_num_nodes = 1; // MHA
#endif

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHATransposedB,
    MHATransposedB,
    ::testing::Combine(::testing::ValuesIn(inputShapesTransposedB),
                       ::testing::Values(std::vector<element::Type>{}),
                       ::testing::Values(ov::element::f32),
                       ::testing::ValuesIn({true}),  // Need to support False for graph builder in tests
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(expected_num_nodes),
                       ::testing::Values(1),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

std::vector<std::vector<ov::test::InputShape>> shared_kv_shapes = {{
    {PartialShape{-1, -1, -1, -1}, {{1, 3, 64, 128}}},
    {PartialShape{-1, -1, -1, -1}, {{1, 3, 64, 128}}},
}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA_SharedKV,
                         MHASharedKV,
                         ::testing::Combine(::testing::ValuesIn(shared_kv_shapes),
                                            ::testing::ValuesIn(precision_f32(2)),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(false),
                                            ::testing::Values(MHA::default_thread_count),
                                            ::testing::Values(expected_num_nodes),
                                            ::testing::Values(1),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(CPUTestUtils::empty_plugin_config)),
                         MHA::getTestCaseName);
}  // namespace
}  // namespace snippets
}  // namespace test
}  // namespace ov
