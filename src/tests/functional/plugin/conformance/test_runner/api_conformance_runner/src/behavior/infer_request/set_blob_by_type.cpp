// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/set_blob_by_type.hpp"
#include "common_test_utils/test_constants.hpp"
#include "api_conformance_helpers.hpp"

namespace {
using namespace ov::test::conformance;
using namespace BehaviorTestsDefinitions;

const std::vector<FuncTestUtils::BlobType> setBlobTypes = {
        FuncTestUtils::BlobType::Compound,
        FuncTestUtils::BlobType::Batched,
        FuncTestUtils::BlobType::Memory,
//    FuncTestUtils::BlobType::Remote,
        FuncTestUtils::BlobType::I420,
        FuncTestUtils::BlobType::NV12
};

const std::map<std::string, std::string> ConfigBlobType{}; //nothing special

INSTANTIATE_TEST_SUITE_P(smoke_Behavior, InferRequestSetBlobByType,
                         ::testing::Combine(::testing::ValuesIn(setBlobTypes),
                                            ::testing::Values(targetDevice),
                                            ::testing::Values(ConfigBlobType)),
                         InferRequestSetBlobByType::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_Behavior_Multi, InferRequestSetBlobByType,
                         ::testing::Combine(::testing::ValuesIn(setBlobTypes),
                                            ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                            ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_MULTI))),
                         InferRequestSetBlobByType::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_Auto, InferRequestSetBlobByType,
                         ::testing::Combine(::testing::ValuesIn(setBlobTypes),
                                            ::testing::Values(CommonTestUtils::DEVICE_AUTO + std::string(":") + targetDevice),
                                            ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_AUTO))),
                         InferRequestSetBlobByType::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_Hetero, InferRequestSetBlobByType,
                         ::testing::Combine(::testing::ValuesIn(setBlobTypes),
                                            ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                                            ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_HETERO))),
                         InferRequestSetBlobByType::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_Batch, InferRequestSetBlobByType,
                         ::testing::Combine(::testing::ValuesIn(setBlobTypes),
                                            ::testing::Values(CommonTestUtils::DEVICE_BATCH),
                                            ::testing::ValuesIn(generate_configs(CommonTestUtils::DEVICE_BATCH))),
                         InferRequestSetBlobByType::getTestCaseName);
} // namespace
