// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/msda_pattern.hpp"

namespace {

using ov::test::MSDAPattern;
using ov::test::MSDAShapes;

INSTANTIATE_TEST_SUITE_P(smoke_MSDAPattern,
                         MSDAPattern,
                         ::testing::Combine(::testing::Values(
                                                // Deformable-DETR encoder layer at 800x1333, queries are the keys.
                                                MSDAShapes{{{100, 167}, {50, 84}, {25, 42}, {13, 21}}, 1, 22223, 8, 32, 4},
                                                MSDAShapes{{{8, 8}, {4, 4}}, 2, 10, 2, 16, 2}),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         MSDAPattern::getTestCaseName);

}  // namespace
