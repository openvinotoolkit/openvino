// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/lora_pattern.hpp"

using namespace ov::test;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke,
                         LoraPatternConvolution,
                         ::testing::Values(ov::test::utils::DEVICE_CPU),
                         LoraPatternBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke,
                         LoraPatternMatmul,
                         ::testing::Values(ov::test::utils::DEVICE_CPU),
                         LoraPatternBase::getTestCaseName);

}  // namespace
