// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/set_blob_by_type.hpp"
#include "common_test_utils/test_constants.hpp"
#include "conformance.hpp"

namespace {

using namespace ConformanceTests;
using namespace BehaviorTestsDefinitions;

const std::vector<FuncTestUtils::BlobType> BlobTypes = {
        FuncTestUtils::BlobType::Compound,
        FuncTestUtils::BlobType::Batched,
        FuncTestUtils::BlobType::Memory,
//    FuncTestUtils::BlobType::Remote,
        FuncTestUtils::BlobType::I420,
        FuncTestUtils::BlobType::NV12
};

const std::map<std::string, std::string> ConfigBlobType{}; //nothing special
const std::map<std::string, std::string> autoConfigBlobType{};
const std::map<std::string, std::string> multiConfigBlobType{{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), targetDevice}};
const std::map<std::string, std::string> heteroConfigBlobType{{"TARGET_FALLBACK", targetDevice}};

INSTANTIATE_TEST_SUITE_P(smoke_Behavior, InferRequestSetBlobByType,
                         ::testing::Combine(::testing::ValuesIn(BlobTypes),
                                            ::testing::Values(targetDevice),
                                            ::testing::Values(ConfigBlobType)),
                         InferRequestSetBlobByType::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_Behavior_Multi, InferRequestSetBlobByType,
                         ::testing::Combine(::testing::ValuesIn(BlobTypes),
                                            ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::Values(multiConfigBlobType)),
                         InferRequestSetBlobByType::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_Auto, InferRequestSetBlobByType,
                         ::testing::Combine(::testing::ValuesIn(BlobTypes),
                                            ::testing::Values(CommonTestUtils::DEVICE_AUTO + std::string(":") + targetDevice),
                                            ::testing::Values(autoConfigBlobType)),
                         InferRequestSetBlobByType::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_Hetero, InferRequestSetBlobByType,
                         ::testing::Combine(::testing::ValuesIn(BlobTypes),
                                            ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                            ::testing::Values(heteroConfigBlobType)),
                         InferRequestSetBlobByType::getTestCaseName);
} // namespace
