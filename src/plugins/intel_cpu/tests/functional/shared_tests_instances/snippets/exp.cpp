// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/exp.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {
// ===================================Exp=========================================================//
std::vector<ov::test::InputShape> inShapes{
    {PartialShape{}, {{1, 1, 32, 128}}},
    {PartialShape{-1, -1, -1}, {{1, 32, 128}, {1, 32, 30}, {1, 32, 1}}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Exp, Exp,
                        ::testing::Combine(
                            ::testing::ValuesIn(inShapes),
                            ::testing::Values(ov::element::f32),
                            ::testing::Values(1), // Exp
                            ::testing::Values(1),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        Exp::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ExpReciprocal, ExpReciprocal,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inShapes),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(1), // Exp
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         Exp::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov