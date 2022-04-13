// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/set_blob_by_type.hpp"
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

INSTANTIATE_TEST_SUITE_P(ie_infer_request, InferRequestSetBlobByType,
                         ::testing::Combine(::testing::ValuesIn(setBlobTypes),
                                            ::testing::ValuesIn(return_all_possible_device_combination()),
                                            ::testing::ValuesIn(empty_config)),
                         InferRequestSetBlobByType::getTestCaseName);
} // namespace
