// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/eltwise_two_results.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {
namespace {

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, EltwiseTwoResults,
                        ::testing::Combine(
                             ::testing::Values(ov::Shape {1, 64, 10, 10}),
                             ::testing::Values(ov::Shape {1, 64, 10,  1}),
                             ::testing::Values(2),
                             ::testing::Values(2),
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         EltwiseTwoResults::getTestCaseName);

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov