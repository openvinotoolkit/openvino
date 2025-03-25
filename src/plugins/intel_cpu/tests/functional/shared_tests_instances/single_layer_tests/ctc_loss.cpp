// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/ctc_loss.hpp"

namespace {
using ov::test::CTCLossLayerTest;

const std::vector<ov::element::Type> f_type = {
        ov::element::f32,
        ov::element::f16
};
const std::vector<ov::element::Type> i_type = {
        ov::element::i32,
        ov::element::i64
};

const std::vector<bool> preprocessCollapseRepeated = {true, false};
const std::vector<bool> ctcMergeRepeated = {true, false};
const std::vector<bool> unique = {true, false};

const auto ctcLossArgsSubset1 = ::testing::Combine(
        ::testing::ValuesIn(std::vector<std::vector<int>>({{2, 3}, {3, 3}})), // logits length
        ::testing::ValuesIn(std::vector<std::vector<std::vector<int>>>(
            {{{0, 1, 0}, {1, 0, 1}}, {{0, 1, 2}, {1, 1, 1}}})),               // labels
        ::testing::ValuesIn(std::vector<std::vector<int>>({{2, 2}, {2, 1}})), // labels length
        ::testing::Values(2),                                                 // blank index
        ::testing::ValuesIn(preprocessCollapseRepeated),
        ::testing::ValuesIn(ctcMergeRepeated),
        ::testing::ValuesIn(unique)
);

INSTANTIATE_TEST_SUITE_P(smoke_Set1, CTCLossLayerTest,
                        ::testing::Combine(
                            ctcLossArgsSubset1,
                            ::testing::Values(ov::test::static_shapes_to_test_representation({{2, 3, 3}})),
                            ::testing::ValuesIn(f_type),
                            ::testing::ValuesIn(i_type),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        CTCLossLayerTest::getTestCaseName);

const auto ctcLossArgsSubset2 = ::testing::Combine(
        ::testing::ValuesIn(std::vector<std::vector<int>>({{6, 5, 6}, {5, 5, 5}})), // logits length
        ::testing::ValuesIn(std::vector<std::vector<std::vector<int>>>(
            {{{4, 1, 2, 3, 4, 5}, {5, 4, 3, 0, 1, 0}, {2, 1, 3, 1, 3, 0}},
             {{2, 1, 5, 3, 2, 6}, {3, 3, 3, 3, 3, 3}, {6, 5, 6, 5, 6, 5}}})),       // labels
        ::testing::ValuesIn(std::vector<std::vector<int>>({{4, 3, 5}, {3, 3, 5}})), // labels length
        ::testing::ValuesIn(std::vector<int>({0, 7})),                              // blank index
        ::testing::ValuesIn(preprocessCollapseRepeated),
        ::testing::ValuesIn(ctcMergeRepeated),
        ::testing::ValuesIn(unique)
);

INSTANTIATE_TEST_SUITE_P(smoke_Set2, CTCLossLayerTest,
                        ::testing::Combine(
                            ctcLossArgsSubset2,
                            ::testing::Values(ov::test::static_shapes_to_test_representation({{3, 6, 8}})),
                            ::testing::ValuesIn(f_type),
                            ::testing::ValuesIn(i_type),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        CTCLossLayerTest::getTestCaseName);
}  // namespace
