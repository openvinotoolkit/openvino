// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/reduce.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

const std::vector<InputShape> inputShape = {
    {{}, {{1, 3, 128, 128}}},
    {{}, {{1, 3, 128, 15}}},
    {{}, {{1, 3, 15, 16}}},
    {{-1, -1, -1, -1}, {{1, 3, 128, 128}, {1, 3, 128, 15}, {1, 3, 15, 16}}}
};

const std::vector<ov::test::utils::ReductionType> reduce_types = {ov::test::utils::ReductionType::Max,
                                                                  ov::test::utils::ReductionType::Sum};

const std::vector<std::vector<int>> axes = {{3}, {-1}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Reduce, Reduce,
                         ::testing::Combine(::testing::ValuesIn(inputShape),
                                            ::testing::ValuesIn(reduce_types),
                                            ::testing::ValuesIn(axes),
                                            ::testing::Values(true),
                                            ::testing::Values(1),
                                            ::testing::Values(1),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         Reduce::getTestCaseName);
} // namespace
} // namespace snippets
} // namespace test
} // namespace ov