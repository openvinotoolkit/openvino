// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/search_sorted.hpp"

#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace SearchSorted {
INSTANTIATE_TEST_SUITE_P(smoke_SearchSortedLayoutTestF32,
                         SearchSortedLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(SearchSortedParamsVector),
                                            testing::Values(ElementType::f32)),
                         SearchSortedLayerCPUTest::getTestCaseName);
}  // namespace SearchSorted
}  // namespace test
}  // namespace ov
