// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/plugin/set_preprocess.hpp"

using namespace BehaviorTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {{}};

const std::vector<std::map<std::string, std::string>> multiConfigs = {
    {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_TEMPLATE}}};

const std::vector<std::map<std::string, std::string>> heteroConfigs = {
    {{"TARGET_FALLBACK", ov::test::utils::DEVICE_TEMPLATE}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferRequestPreprocessTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(configs)),
                         InferRequestPreprocessTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         InferRequestPreprocessTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         InferRequestPreprocessTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         InferRequestPreprocessTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(heteroConfigs)),
                         InferRequestPreprocessTest::getTestCaseName);

const std::vector<InferenceEngine::Precision> ioPrecisions = {InferenceEngine::Precision::FP32,
                                                              InferenceEngine::Precision::U8};
const std::vector<InferenceEngine::Layout> netLayouts = {
    InferenceEngine::Layout::NCHW,
};

const std::vector<InferenceEngine::Layout> ioLayouts = {
    InferenceEngine::Layout::NCHW,
    InferenceEngine::Layout::NHWC,
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferRequestPreprocessConversionTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(ioPrecisions),
                                            ::testing::ValuesIn(ioPrecisions),
                                            ::testing::ValuesIn(netLayouts),
                                            ::testing::ValuesIn(ioLayouts),
                                            ::testing::ValuesIn(ioLayouts),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                                            ::testing::ValuesIn(configs)),
                         InferRequestPreprocessConversionTest::getTestCaseName);

}  // namespace
