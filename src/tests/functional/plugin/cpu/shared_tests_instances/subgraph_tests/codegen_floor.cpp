
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/codegen_floor.hpp"

using namespace LayerTestsDefinitions;

namespace {

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32
    };

    INSTANTIATE_TEST_SUITE_P(NoReshape, CodegenFloor,
            ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(InferenceEngine::SizeVector({1, 3, 16, 16})),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            CodegenFloor::getTestCaseName);
}  // namespace