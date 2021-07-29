// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/layout.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<std::map<std::string, std::string>> configs = {
            {}
    };

    const std::vector<InferenceEngine::Layout> Layout = {
           InferenceEngine::Layout::NCHW,
           InferenceEngine::Layout::NHWC,
           InferenceEngine::Layout::CHW,
           InferenceEngine::Layout::HWC,
           InferenceEngine::Layout::NC,
           InferenceEngine::Layout::C
    };

    const std::vector<std::vector<size_t>> inputShapes = {
            { 1, 3, 16, 16 },
            { 3, 32, 16 },
            { 1, 3 },
            { 3 }
    };

    INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, LayoutTest,
            ::testing::Combine(
                    ::testing::Values(InferenceEngine::Precision::FP32),
                    ::testing::Values(CommonTestUtils::DEVICE_CPU, "HETERO:CPU"),
                    ::testing::ValuesIn(configs),
                    ::testing::ValuesIn(Layout),
                    ::testing::ValuesIn(inputShapes)),
            LayoutTest::getTestCaseName);
}  // namespace
