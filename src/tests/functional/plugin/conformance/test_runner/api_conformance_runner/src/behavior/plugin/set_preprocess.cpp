// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/set_preprocess.hpp"
#include "api_conformance_helpers.hpp"

namespace {

using namespace BehaviorTestsDefinitions;
using namespace ov::test::conformance;

const std::vector<InferenceEngine::Precision> netPrecisionsPreprocess = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

INSTANTIATE_TEST_SUITE_P(ie_plugin, InferRequestPreprocessTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisionsPreprocess),
                                ::testing::ValuesIn(return_all_possible_device_combination()),
                                ::testing::ValuesIn(empty_config)),
                         InferRequestPreprocessTest::getTestCaseName);

const std::vector<InferenceEngine::Precision> ioPrecisionsPreprocess = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::U8
};
const std::vector<InferenceEngine::Layout> netLayoutsPreprocess = {
    InferenceEngine::Layout::NCHW,
    // InferenceEngine::Layout::NHWC
};

const std::vector<InferenceEngine::Layout> ioLayoutsPreprocess = {
    InferenceEngine::Layout::NCHW,
    InferenceEngine::Layout::NHWC
};

INSTANTIATE_TEST_SUITE_P(ie_plugin, InferRequestPreprocessConversionTest,
                    ::testing::Combine(
                            ::testing::ValuesIn(netPrecisionsPreprocess),
                            ::testing::ValuesIn(ioPrecisionsPreprocess),
                            ::testing::ValuesIn(ioPrecisionsPreprocess),
                            ::testing::ValuesIn(netLayoutsPreprocess),
                            ::testing::ValuesIn(ioLayoutsPreprocess),
                            ::testing::ValuesIn(ioLayoutsPreprocess),
                            ::testing::Bool(),
                            ::testing::Bool(),
                            ::testing::ValuesIn(return_all_possible_device_combination()),
                            ::testing::ValuesIn(empty_config)),
                    InferRequestPreprocessConversionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ie_plugin, InferRequestPreprocessDynamicallyInSetBlobTest,
                    ::testing::Combine(
                            ::testing::ValuesIn(netPrecisionsPreprocess),
                            ::testing::Bool(),
                            ::testing::Bool(),
                            ::testing::ValuesIn(netLayoutsPreprocess),
                            ::testing::Bool(),
                            ::testing::Bool(),
                            ::testing::Values(true), // only SetBlob
                            ::testing::Values(true), // only SetBlob
                            ::testing::ValuesIn(return_all_possible_device_combination()),
                            ::testing::ValuesIn(empty_config)),
                    InferRequestPreprocessDynamicallyInSetBlobTest::getTestCaseName);
}  // namespace
