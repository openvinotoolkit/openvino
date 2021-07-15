// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior2/infer_request/set_blob_by_type.hpp"
#include "common_test_utils/test_constants.hpp"
#include "conformance.hpp"

namespace ConformanceTests {
using namespace BehaviorTestsDefinitions;

namespace {
const std::vector<FuncTestUtils::BlobType> BlobTypes = {
        FuncTestUtils::BlobType::Compound,
        FuncTestUtils::BlobType::Batched,
        FuncTestUtils::BlobType::Memory,
//    FuncTestUtils::BlobType::Remote,
        FuncTestUtils::BlobType::I40,
        FuncTestUtils::BlobType::NV12
};

const std::map<std::string, std::string> Config{}; //nothing special
const std::map<std::string, std::string> autoConfig{};
const std::map<std::string, std::string> multiConfig{{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), targetDevice}};
const std::map<std::string, std::string> heteroConfig{{"TARGET_FALLBACK", targetDevice}};

INSTANTIATE_TEST_SUITE_P(smoke_Behavior, InferRequestSetBlobByType,
                         ::testing::Combine(::testing::ValuesIn(BlobTypes),
                                            ::testing::Values(targetDevice),
                                            ::testing::Values(Config)),
                         InferRequestSetBlobByType::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_Behavior_Multi, InferRequestSetBlobByType,
                         ::testing::Combine(::testing::ValuesIn(BlobTypes),
                                            ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::Values(multiConfig)),
                         InferRequestSetBlobByType::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_Auto, InferRequestSetBlobByType,
                         ::testing::Combine(::testing::ValuesIn(BlobTypes),
                                            ::testing::Values(CommonTestUtils::DEVICE_AUTO + std::string(":") + targetDevice),
                                            ::testing::Values(autoConfig)),
                         InferRequestSetBlobByType::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_Hetero, InferRequestSetBlobByType,
                         ::testing::Combine(::testing::ValuesIn(BlobTypes),
                                            ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                            ::testing::Values(heteroConfig)),
                         InferRequestSetBlobByType::getTestCaseName);
} // namespace
} // namespace ConformanceTests