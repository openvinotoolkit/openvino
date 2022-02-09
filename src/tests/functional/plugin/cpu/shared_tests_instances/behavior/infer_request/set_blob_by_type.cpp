// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/set_blob_by_type.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace BehaviorTestsDefinitions;
using namespace InferenceEngine;

const std::vector<FuncTestUtils::BlobType> BlobTypes = {
    FuncTestUtils::BlobType::Compound,
    FuncTestUtils::BlobType::Batched,
    FuncTestUtils::BlobType::Memory,
//    FuncTestUtils::BlobType::Remote,
    FuncTestUtils::BlobType::I420,
    FuncTestUtils::BlobType::NV12
};

const std::map<std::string, std::string> cpuConfig{}; //nothing special
const std::map<std::string, std::string> autoConfig{};
const std::map<std::string, std::string> multiConfig{{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , CommonTestUtils::DEVICE_CPU}};
const std::map<std::string, std::string> heteroConfig{{ "TARGET_FALLBACK", CommonTestUtils::DEVICE_CPU }};

INSTANTIATE_TEST_SUITE_P(smoke_Behavior, InferRequestSetBlobByType,
    ::testing::Combine(::testing::ValuesIn(BlobTypes),
                       ::testing::Values(CommonTestUtils::DEVICE_CPU),
                       ::testing::Values(cpuConfig)),
    InferRequestSetBlobByType::getTestCaseName);


INSTANTIATE_TEST_SUITE_P(smoke_Behavior_Multi, InferRequestSetBlobByType,
    ::testing::Combine(::testing::ValuesIn(BlobTypes),
                       ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                       ::testing::Values(multiConfig)),
    InferRequestSetBlobByType::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_Auto, InferRequestSetBlobByType,
    ::testing::Combine(::testing::ValuesIn(BlobTypes),
                       ::testing::Values(CommonTestUtils::DEVICE_AUTO + std::string(":") + CommonTestUtils::DEVICE_CPU),
                       ::testing::Values(autoConfig)),
    InferRequestSetBlobByType::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_Hetero, InferRequestSetBlobByType,
    ::testing::Combine(::testing::ValuesIn(BlobTypes),
                       ::testing::Values(CommonTestUtils::DEVICE_HETERO),
                       ::testing::Values(heteroConfig)),
    InferRequestSetBlobByType::getTestCaseName);
