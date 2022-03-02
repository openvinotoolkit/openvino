
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/codegen_quantized.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32
};

const std::vector<ov::element::Type> inputPrecisions = {
    ov::element::f32
};

INSTANTIATE_TEST_SUITE_P(CodeGeneration, CodegenQuantized,
    ::testing::Combine(
    ::testing::ValuesIn(netPrecisions),
    ::testing::Values(InferenceEngine::SizeVector({1, 3, 1024 * 4, 1024 * 4})),
    ::testing::ValuesIn(inputPrecisions),
    ::testing::ValuesIn({ false, true }),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    CodegenQuantized::getTestCaseName);
}  // namespace
