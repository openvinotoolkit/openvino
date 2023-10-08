// Copyright (C) 2018-2023 Intel Corporation
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
};

const std::map<std::string, std::string> autoConfig{
    {MULTI_CONFIG_KEY(DEVICE_PRIORITIES), ov::test::utils::DEVICE_TEMPLATE}};

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_Multi,
                         InferRequestSetBlobByType,
                         ::testing::Combine(::testing::ValuesIn(BlobTypes),
                                            ::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::Values(autoConfig)),
                         InferRequestSetBlobByType::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_Auto,
                         InferRequestSetBlobByType,
                         ::testing::Combine(::testing::ValuesIn(BlobTypes),
                                            ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::Values(autoConfig)),
                         InferRequestSetBlobByType::getTestCaseName);
