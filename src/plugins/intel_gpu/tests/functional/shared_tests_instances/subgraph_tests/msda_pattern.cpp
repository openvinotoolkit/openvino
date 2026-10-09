// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/msda_pattern.hpp"

namespace {

using ov::test::MSDAForm;
using ov::test::MSDAPattern;
using ov::test::MSDAShapes;

const std::vector<MSDAShapes> shapes = {
    // GroundingDINO levels at 800x1333 with fewer queries than the 900 of the decoder.
    {{{100, 167}, {50, 84}, {25, 42}, {13, 21}}, 1, 50, 8, 32, 4},
    // Encoder style: the queries are the keys.
    {{{8, 10}, {4, 5}}, 1, 100, 2, 16, 4},
    {{{4, 5}, {6, 6}, {8, 7}}, 1, 6, 2, 8, 2},
    {{{6, 6}, {3, 3}}, 2, 7, 4, 16, 2},
};

INSTANTIATE_TEST_SUITE_P(smoke_MSDAPattern,
                         MSDAPattern,
                         ::testing::Combine(::testing::ValuesIn(shapes),
                                            ::testing::Values(MSDAForm::VariadicSplit, MSDAForm::StridedSlice),
                                            ::testing::Values(ov::element::f32, ov::element::f16),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         MSDAPattern::getTestCaseName);

}  // namespace
