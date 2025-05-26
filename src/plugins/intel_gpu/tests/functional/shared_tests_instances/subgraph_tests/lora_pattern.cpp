// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/lora_pattern.hpp"

using namespace ov::test;

namespace {

const std::vector<size_t> M = { 1, 20, 256 };
const std::vector<size_t> N = { 2048 };
const std::vector<size_t> K = { 512, 563 };
const std::vector<size_t> lora_rank = { 16, 25, 64, 128 };

INSTANTIATE_TEST_SUITE_P(smoke,
                         LoraPatternMatmul,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::ValuesIn(M),
                                            ::testing::ValuesIn(N),
                                            ::testing::ValuesIn(K),
                                            ::testing::ValuesIn(lora_rank)),
                         LoraPatternMatmul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke,
                         LoraPatternConvolution,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(64),
                                            ::testing::Values(25)),
                         LoraPatternConvolution::getTestCaseName);

}  // namespace
