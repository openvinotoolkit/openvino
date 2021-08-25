// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/infer_request/io_blob.hpp"
#include "ie_plugin_config.hpp"
#include "conformance.hpp"

namespace {
using namespace BehaviorTestsDefinitions;
using namespace ConformanceTests;

const std::vector<std::map<std::string, std::string>> configsIOBlob = {
        {},
};

const std::vector<std::map<std::string, std::string>> MulticonfigsIOBlob = {
        {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES), targetDevice }}
};

const std::vector<std::map<std::string, std::string>> AutoconfigsIOBlob = {
        {{ AUTO_CONFIG_KEY(DEVICE_LIST), targetDevice}}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestIOBBlobTest,
                        ::testing::Combine(
                                ::testing::Values(targetDevice),
                                ::testing::ValuesIn(configsIOBlob)),
                         InferRequestIOBBlobTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestIOBBlobTest,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                ::testing::ValuesIn(MulticonfigsIOBlob)),
                         InferRequestIOBBlobTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestIOBBlobTest,
                        ::testing::Combine(
                                ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                ::testing::ValuesIn(AutoconfigsIOBlob)),
                         InferRequestIOBBlobTest::getTestCaseName);
}  // namespace
