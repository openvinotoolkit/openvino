// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/transpose_softmax.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

const std::vector<std::vector<InputShape>> inputShape = {
    {{{}, {{1, 128, 3, 16}}}},
    {{{-1, -1, -1, -1}, {{1, 128, 3, 16}, {1, 64, 3, 8}, {1, 128, 3, 16}}}},
    {{{-1, 100, -1, -1}, {{1, 100, 3, 16}, {1, 100, 3, 8}, {1, 100, 2, 16}, {1, 100, 3, 8}}}},
    {{{-1, -1, -1, 16}, {{1, 128, 3, 16}, {1, 32, 3, 16}, {1, 32, 3, 16}, {1, 100, 2, 16}}}},
};

const std::vector<std::vector<InputShape>> inputShapeWithEltwise = {
    {{{}, {{1, 128, 3, 16}}},
     {{}, {{1, 3, 16, 128}}}},
    {{{-1, -1, -1, -1}, {{1, 128, 3, 16}, {1, 64, 3, 8}, {1, 128, 3, 16}}},
     {{-1, -1, -1, -1}, {{1, 3, 16, 128}, {1, 3, 8, 64}, {1, 3, 16, 128}}}},
    {{{-1, 100, -1, -1}, {{1, 100, 3, 16}, {1, 100, 3, 8}, {1, 100, 2, 16}, {1, 100, 3, 8}}},
     {{-1, -1, -1, 100}, {{1, 1, 1, 100}, {1, 3, 8, 100}, {1, 2, 16, 100}, {1, 3, 8, 100}}}},
    {{{-1, 100, -1, 3}, {{1, 100, 3, 3}, {1, 100, 3, 3}, {1, 100, 2, 3}, {1, 100, 3, 3}}},
     {{-1, -1, -1, 100}, {{1, 1, 1, 100}, {1, 3, 3, 100}, {1, 2, 3, 100}, {1, 1, 1, 100}}}},
    {{{-1, -1, -1, 16}, {{1, 128, 3, 16}, {1, 32, 3, 16}, {1, 32, 3, 16}, {1, 100, 2, 16}}},
     {{-1, -1, 16, -1}, {{1, 3, 16, 128}, {1, 1, 16, 32}, {1, 3, 16, 32}, {1, 2, 16, 100}}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_TransposeSoftmax, TransposeSoftmax,
                     ::testing::Combine(
                             ::testing::ValuesIn(inputShape),
                             ::testing::Values(std::vector<int64_t>{0, 2, 3, 1}),
                             ::testing::Values(-1),
                             ::testing::Values(1),
                             ::testing::Values(1),
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                     TransposeSoftmax::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_TransposeSoftmaxEltwise, TransposeSoftmaxEltwise,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapeWithEltwise),
                                 ::testing::Values(std::vector<int64_t>{0, 2, 3, 1}),
                                 ::testing::Values(-1),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeSoftmax::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov