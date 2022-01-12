
// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "snippets/codegen_bert.hpp"
#include "common_test_utils/test_constants.hpp"
//  todo: Rewrite this test using Snippets test infrastructure. See add_convert or conv_eltwise for example
using namespace LayerTestsDefinitions;

namespace {

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32
    };

    INSTANTIATE_TEST_SUITE_P(NoReshape, CodegenBert,
            ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(InferenceEngine::SizeVector({1, 42, 16, 64})),
            ::testing::Values(InferenceEngine::SizeVector({1, 42, 64, 64})),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            CodegenBert::getTestCaseName);
}  // namespace