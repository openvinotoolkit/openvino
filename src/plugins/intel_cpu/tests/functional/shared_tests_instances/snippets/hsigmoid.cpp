// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/unary_activation.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

std::vector<ov::test::InputShape> inShapes{
    {PartialShape{}, {{1, 1, 32, 128}}},
    {PartialShape{-1, -1, -1}, {{1, 32, 128}, {1, 32, 30}, {1, 32, 1}}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_HSigmoid, HSigmoid,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inShapes),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(1), // HSigmoid
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         UnaryActivation::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov
