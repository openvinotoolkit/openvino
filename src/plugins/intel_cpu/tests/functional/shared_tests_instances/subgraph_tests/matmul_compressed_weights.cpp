// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/matmul_compressed_weights.hpp"

#include <vector>

using namespace ov::test;

namespace {
const std::vector<ov::element::Type> input_precision = {ov::element::f32};
const std::vector<ov::element::Type> weights_precision = {ov::element::i4,
                                                          ov::element::u4,
                                                          ov::element::i8,
                                                          ov::element::u8};

const std::vector<ov::AnyMap> configs = {{}};

std::vector<ov::Shape> input_shapes = {{32, 10, 32}, {1, 5, 32}, {5, 32}};

std::vector<ov::Shape> weights_shapes = {{32, 128, 32}, {32, 64, 32}};

INSTANTIATE_TEST_SUITE_P(smoke_MatmulCompressedWeights,
                         MatmulCompressedTest,
                         ::testing::Combine(::testing::ValuesIn(input_precision),
                                            ::testing::ValuesIn(weights_precision),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(weights_shapes)),
                         MatmulCompressedTest::getTestCaseName);
}  // namespace
