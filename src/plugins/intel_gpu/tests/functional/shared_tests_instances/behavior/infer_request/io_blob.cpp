// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/infer_request/io_blob.hpp"
#include "ie_plugin_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
auto configs = []() {
    return std::vector<std::map<std::string, std::string>>{
        {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_GPU}}};
};

auto autoconfigs = []() {
    return std::vector<std::map<std::string, std::string>>{
        {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_GPU}},
        {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
          std::string(ov::test::utils::DEVICE_CPU) + "," + ov::test::utils::DEVICE_GPU}}};
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferRequestIOBBlobTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         InferRequestIOBBlobTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         InferRequestIOBBlobTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(configs())),
                         InferRequestIOBBlobTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         InferRequestIOBBlobTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoconfigs())),
                         InferRequestIOBBlobTest::getTestCaseName);

}  // namespace
