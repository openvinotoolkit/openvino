
// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/codegen_convert.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<ov::element::Type> inTypes = {
        ov::element::f32,
        ov::element::bf16,
        ov::element::i8,
        ov::element::u8,
};

const std::vector<InferenceEngine::SizeVector> inputShapes = {
        InferenceEngine::SizeVector({8, 8, 8}),
        InferenceEngine::SizeVector({16, 28, 1}),
        InferenceEngine::SizeVector({2, 17, 5}),
};

INSTANTIATE_TEST_SUITE_P(smoke_CodegenConvert, CodegenConvert,
        ::testing::Combine(
        ::testing::ValuesIn(inTypes),
        ::testing::ValuesIn(inputShapes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        CodegenConvert::getTestCaseName);
}  // namespace