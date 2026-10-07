// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/msda_grid_sample_pattern.hpp"

namespace {

using ov::test::MSDAGridSamplePattern;
using ov::test::MSDAShapes;

INSTANTIATE_TEST_SUITE_P(smoke_MSDAGridSamplePattern,
                         MSDAGridSamplePattern,
                         ::testing::Combine(::testing::Values(MSDAShapes{{{4, 5}, {6, 6}, {8, 7}}, 1, 6, 2, 8, 2},
                                                              // GroundingDINO decoder geometry with fewer queries.
                                                              MSDAShapes{{{20, 34}, {10, 17}, {5, 9}, {3, 5}}, 1, 50, 8, 32, 4},
                                                              MSDAShapes{{{6, 6}, {3, 3}}, 2, 7, 4, 16, 2}),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         MSDAGridSamplePattern::getTestCaseName);

}  // namespace
