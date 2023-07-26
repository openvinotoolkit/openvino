// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/select.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Select, Select,
        ::testing::Combine(
                ::testing::ValuesIn({ov::Shape{1, 5, 5, 35}, }),
                ::testing::ValuesIn({ov::Shape{1, 5, 5, 35}, }),
                ::testing::ValuesIn({ov::Shape{1}}),
                ::testing::ValuesIn({ov::element::f32, ov::element::i8}),
                ::testing::Values(1),
                ::testing::Values(1),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
        Select::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_BroadcastSelect, BroadcastSelect,
                         ::testing::Combine(
                                 ::testing::ValuesIn({Shape{1, 8, 2, 1}, Shape{1, 1, 1, 1}}),
                                 ::testing::ValuesIn({Shape{1, 8, 2, 10}, Shape{1, 8, 2, 1}}),
                                 ::testing::ValuesIn({Shape{1, 8, 2, 10}, Shape{1, 1, 1, 1}}),
                                 ::testing::ValuesIn({Shape{1, 8, 2, 1}, Shape{1, 8, 2, 10}}),
                                 ::testing::ValuesIn({ov::element::f32, ov::element::i8}),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         BroadcastSelect::getTestCaseName);


}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov