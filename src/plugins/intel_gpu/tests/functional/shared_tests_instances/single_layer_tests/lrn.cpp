// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/lrn.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::LrnLayerTest;

const std::vector<ov::element::Type> netPrecisions = {ov::element::f32,
                                                      ov::element::f16};

const std::vector<std::vector<int64_t>> axes = {{1}, {2, 3}};

const double alpha = 9.9e-05;
const double beta = 2;
const double bias = 1.0;
const size_t size = 5;

INSTANTIATE_TEST_SUITE_P(smoke_LrnCheck, LrnLayerTest,
                        ::testing::Combine(::testing::Values(alpha),
                                           ::testing::Values(beta),
                                           ::testing::Values(bias),
                                           ::testing::Values(size),
                                           ::testing::ValuesIn(axes),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>(
                                                                {{{10, 10, 3, 2}}}))),
                                           ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        LrnLayerTest::getTestCaseName);

}  // namespace
