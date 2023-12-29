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
// These  inputs are needed to test static Loop optimizations (emit the whole tile, body with increments, set WA etc)
std::vector<ov::test::InputShape> inShapesStatic{{{}, {{1, 1, 32, 128}}}};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, Exp,
                        ::testing::Combine(
                            ::testing::ValuesIn(inShapesStatic),
                            ::testing::Values(ov::element::f32),
                            ::testing::Values(1), // Exp
                            ::testing::Values(1),
                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        Exp::getTestCaseName);

// DS
// std::vector<InputShape> inShapesDynamic1{
//         {
//         {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
//         {{1, 3, 1, 10}, {1, 3, 10, 10}, {1, 3, 1, 10}}},
//         }
// };
// std::vector<InputShape> inShapesDynamic2{
//         {
//         {{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension::dynamic()},
//         {{1, 3, 10, 1}, {1, 3, 1, 1}, {1, 3, 10, 1}}},
//         }
// };
//INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise_Add, Add,
//                         ::testing::Combine(
//                                 ::testing::ValuesIn(inShapesDynamic1),
//                                 ::testing::ValuesIn(inShapesDynamic2),
//                                 ::testing::Values(ov::element::f32),
//                                 ::testing::Values(1),
//                                 ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
//                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
//                         Add::getTestCaseName);


} // namespace
} // namespace snippets
} // namespace test
} // namespace ov