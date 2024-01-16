// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/io_blob.hpp"

#include <vector>

#include "ie_plugin_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<std::map<std::string, std::string>> Autoconfigs = {
    {{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), ov::test::utils::DEVICE_TEMPLATE}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         InferRequestIOBBlobTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(Autoconfigs)),
                         InferRequestIOBBlobTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         InferRequestIOBBlobTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(Autoconfigs)),
                         InferRequestIOBBlobTest::getTestCaseName);

}  // namespace
