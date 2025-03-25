// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/search_sorted.hpp"

namespace ov {
namespace test {

INSTANTIATE_TEST_SUITE_P(smoke_SearchSortedTest,
                         SearchSortedLayerTest,
                         ::testing::Combine(::testing::ValuesIn(SearchSortedLayerTest::GenerateParams()),
                                            testing::Values(ElementType::f32, ElementType::f16, ElementType::i64, ElementType::u32),
                                            testing::Values(ov::test::utils::DEVICE_CPU)),
                         SearchSortedLayerTest::getTestCaseName);

}  // namespace test
}  // namespace ov
