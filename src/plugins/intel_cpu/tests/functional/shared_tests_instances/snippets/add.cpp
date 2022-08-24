// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/add.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, Add,
                     ::testing::Combine(
                             ::testing::Values(ov::Shape {1, 42, 16, 64}),
                             ::testing::Values(ov::Shape {1, 42, 16,  1}),
                             ::testing::Values(1),
                             ::testing::Values(1),
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                     Add::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, AddConst,
                     ::testing::Combine(
                             ::testing::Values(ov::Shape {1, 42, 16, 64}),
                             ::testing::Values(1),
                             ::testing::Values(1),
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                     AddConst::getTestCaseName);

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov