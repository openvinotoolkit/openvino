// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/set_blob_by_type.hpp"
#include "api_conformance_helpers.hpp"

namespace {
using namespace ov::test::conformance;
using namespace BehaviorTestsDefinitions;

const std::vector<ov::test::utils::BlobType> setBlobTypes = {
        ov::test::utils::BlobType::Compound,
        ov::test::utils::BlobType::Batched,
        ov::test::utils::BlobType::Memory,
        ov::test::utils::BlobType::Remote,
};

INSTANTIATE_TEST_SUITE_P(ie_infer_request, InferRequestSetBlobByType,
                         ::testing::Combine(::testing::ValuesIn(setBlobTypes),
                                            ::testing::ValuesIn(return_all_possible_device_combination()),
                                            ::testing::Values(ie_config)),
                         InferRequestSetBlobByType::getTestCaseName);
} // namespace
