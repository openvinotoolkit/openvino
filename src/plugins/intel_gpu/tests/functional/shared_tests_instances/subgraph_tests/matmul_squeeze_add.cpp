// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/matmul_squeeze_add.hpp"

#include <vector>

using namespace ov::test;

namespace {
const std::vector<ov::element::Type> netPrecisions = {ov::element::f32, ov::element::f16};

const std::vector<ov::AnyMap> configs = {{}};

std::vector<ov::Shape> input_shapes = {{1, 8}, {1, 42}, {1, 100}, {1, 128}, {1, 512}};

std::vector<size_t> output_sizes = {1000, 512, 128, 42, 16, 8};

INSTANTIATE_TEST_SUITE_P(MatmulSqueezeAdd,
                         MatmulSqueezeAddTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(output_sizes)),
                         MatmulSqueezeAddTest::getTestCaseName);
}  // namespace
