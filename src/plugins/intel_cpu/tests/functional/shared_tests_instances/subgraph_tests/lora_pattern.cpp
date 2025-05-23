// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/lora_pattern.hpp"

using namespace ov::test;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke,
                         LoraPatternMatmul,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(20),
                                            ::testing::Values(2048),
                                            ::testing::Values(563),
                                            ::testing::Values(25)),
                         LoraPatternMatmul::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke,
                         LoraPatternConvolution,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(64),
                                            ::testing::Values(25)),
                         LoraPatternConvolution::getTestCaseName);

}  // namespace
