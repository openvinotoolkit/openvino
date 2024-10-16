// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/search_sorted.hpp"

namespace ov {
namespace test {

INSTANTIATE_TEST_SUITE_P(
    smoke_SearchSortedTest,
    SearchSortedLayerCPUTest,
    ::testing::Combine(::testing::ValuesIn(SearchSortedParamsVector),
                       testing::Values(ElementType::f32, ElementType::f16, ElementType::i64, ElementType::u32)),
    SearchSortedLayerCPUTest::getTestCaseName);

}  // namespace test
}  // namespace ov
