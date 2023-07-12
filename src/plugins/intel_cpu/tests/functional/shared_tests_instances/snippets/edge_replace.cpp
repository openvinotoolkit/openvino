// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/edge_replace.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

std::vector<ov::PartialShape> input_shapes{{4}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_edge_replace, EdgeReplace,
                         ::testing::Combine(
                                 ::testing::ValuesIn(input_shapes),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(3),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         EdgeReplace::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov