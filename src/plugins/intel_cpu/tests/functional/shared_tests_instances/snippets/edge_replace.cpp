// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/edge_replace.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {

std::vector<InputShape> inShapes0{
        {{}, {{1, 2}}}
};
std::vector<InputShape> inShapes1{
        {{}, {{2}}}
};
std::vector<InputShape> inShapes2{
        {{}, {{1, 2, 2, 3, 2}}}
};
std::vector<InputShape> inShapes3{
        {{}, {{1, 2, 2, 3, 2}}}
};
std::vector<InputShape> inShapes4{
        {{}, {{2, 2, 2, 3, 2}}}
};
std::vector<InputShape> inShapes5{
        {{}, {{2, 2, 2, 3, 2}}}
};
std::vector<InputShape> inShapes6{
        {{}, {{4, 2, 2, 3, 2}}}
};
std::vector<InputShape> inShapes7{
        {{}, {{4, 2, 2, 3, 2}}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_edge_replace, EdgeReplace,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inShapes0),
                                 ::testing::ValuesIn(inShapes1),
                                 ::testing::ValuesIn(inShapes2),
                                 ::testing::ValuesIn(inShapes3),
                                 ::testing::ValuesIn(inShapes4),
                                 ::testing::ValuesIn(inShapes5),
                                 ::testing::ValuesIn(inShapes6),
                                 ::testing::ValuesIn(inShapes7),
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(9),
                                 ::testing::Values(5),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         EdgeReplace::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov