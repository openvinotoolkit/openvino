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

const std::map<std::string, std::string> cpuConfig{}; //nothing special
const std::map<std::string, std::string> heteroConfig{{ "TARGET_FALLBACK", ov::test::utils::DEVICE_CPU }};

INSTANTIATE_TEST_SUITE_P(smoke_Behavior, InferRequestSetBlobByType,
    ::testing::Combine(::testing::ValuesIn(BlobTypes),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::Values(cpuConfig)),
    InferRequestSetBlobByType::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_Hetero, InferRequestSetBlobByType,
    ::testing::Combine(::testing::ValuesIn(BlobTypes),
                       ::testing::Values(ov::test::utils::DEVICE_HETERO),
                       ::testing::Values(heteroConfig)),
    InferRequestSetBlobByType::getTestCaseName);
