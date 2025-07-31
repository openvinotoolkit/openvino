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

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHATransposedB,
    MHATransposedB,
    ::testing::Combine(::testing::ValuesIn(inputShapesTransposedB),
                       ::testing::Values(std::vector<element::Type>{}),
                       ::testing::Values(ov::element::f32),
                       ::testing::ValuesIn({true}),  // Need to support False for graph builder in tests
                       ::testing::Values(MHA::default_thread_count),
                       ::testing::Values(1),
                       ::testing::Values(1),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(CPUTestUtils::empty_plugin_config)),
    MHA::getTestCaseName);

// std::vector<std::vector<ov::test::InputShape>> originalShape_4D_fa {
//     { {{}, {{1, 1, 1024, 256}}}, {{}, {{1, 1, 6400, 256}}}, {{}, {{1, 1, 6400, 256}}} },
//     { {{}, {{1, 1, 1028, 256}}}, {{}, {{1, 1, 6400, 256}}}, {{}, {{1, 1, 6400, 256}}} },
//     { {{}, {{1, 1, 1024, 256}}}, {{}, {{1, 1, 6410, 256}}}, {{}, {{1, 1, 6410, 256}}} },
//     { {{}, {{1, 1, 1028, 256}}}, {{}, {{1, 1, 6410, 256}}}, {{}, {{1, 1, 6410, 256}}} },
//     { {{}, {{1, 1, 1028, 512}}}, {{}, {{1, 1, 6410, 512}}}, {{}, {{1, 1, 6410, 512}}} },
// };

// std::vector<std::vector<ov::test::InputShape>> originalShape_4D_fa {
//     {
//         {PartialShape{-1, -1, -1, -1}, {{1, 1, 1024, 256}, {1, 1, 1028, 256}, {1, 1, 1024, 256}, {1, 1, 1028, 256}}},
//         {PartialShape{-1, -1, -1, -1}, {{1, 1, 6400, 256}, {1, 1, 6410, 256}, {1, 1, 6400, 256}, {1, 1, 6410, 256}}},
//         {PartialShape{-1, -1, -1, -1}, {{1, 1, 6400, 256}, {1, 1, 6410, 256}, {1, 1, 6400, 256}, {1, 1, 6410, 256}}},
//     },
// };

std::vector<std::vector<ov::test::InputShape>> originalShape_4D_fa {
    { {{}, {{1, 1, 1024, 256}}}, {{}, {{1, 1, 6400, 256}}}, {{}, {{1, 1, 6400, 256}}} },
    { {{}, {{1, 1, 1028, 256}}}, {{}, {{1, 1, 6400, 256}}}, {{}, {{1, 1, 6400, 256}}} },
    { {{}, {{1, 1, 1024, 256}}}, {{}, {{1, 1, 6410, 256}}}, {{}, {{1, 1, 6410, 256}}} },
    { {{}, {{1, 1, 1028, 256}}}, {{}, {{1, 1, 6410, 256}}}, {{}, {{1, 1, 6410, 256}}} },
    { {{}, {{1, 1, 1028, 512}}}, {{}, {{1, 1, 6410, 512}}}, {{}, {{1, 1, 6410, 512}}} },
    {
        {PartialShape{-1, -1, -1, -1}, {{1, 1, 4096, 256}, {1, 1, 4096, 256}, {1, 1, 4096, 256}, {1, 1, 4096, 256}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 1, 4100, 256}, {1, 1, 8200, 256}, {1, 1, 12300, 256}, {1, 1, 28736, 256}}},
        {PartialShape{-1, -1, -1, -1}, {{1, 1, 4100, 256}, {1, 1, 8200, 256}, {1, 1, 12300, 256}, {1, 1, 28736, 256}}},
    },
};

// std::vector<std::vector<ov::test::InputShape>> originalShape_4D_fa {
//     { {{}, {{1, 1, 4096, 256}}}, {{}, {{1, 1, 4100, 256}}}, {{}, {{1, 1, 4100, 256}}} },
//     { {{}, {{1, 1, 4096, 256}}}, {{}, {{1, 1, 8200, 256}}}, {{}, {{1, 1, 8200, 256}}} },
//     { {{}, {{1, 1, 4096, 256}}}, {{}, {{1, 1, 12300, 256}}}, {{}, {{1, 1, 12300, 256}}} },
//     { {{}, {{1, 1, 4096, 256}}}, {{}, {{1, 1, 28736, 256}}}, {{}, {{1, 1, 28736, 256}}} },
// };

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_MHATransposedBFA,
    MHATransposedB,
    ::testing::Combine(::testing::ValuesIn(originalShape_4D_fa),
                       ::testing::ValuesIn(precision_f32(3)),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(false),
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
