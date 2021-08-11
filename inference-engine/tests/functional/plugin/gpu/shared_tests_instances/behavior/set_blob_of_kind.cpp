// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/set_blob_of_kind.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace BehaviorTestsDefinitions;
using namespace InferenceEngine;

const std::vector<FuncTestUtils::BlobKind> blobKinds = {
    FuncTestUtils::BlobKind::Simple,
    FuncTestUtils::BlobKind::Compound
    /* BatchOfSimple is not supported on GPU currently. Batch of remote is supported */
    /* , FuncTestUtils::BlobKind::BatchOfSimple */
};

const SetBlobOfKindConfig gpuConfig{}; //nothing special

INSTANTIATE_TEST_SUITE_P(smoke_SetBlobOfKindGPU, SetBlobOfKindTest,
    ::testing::Combine(::testing::ValuesIn(blobKinds),
                       ::testing::Values(CommonTestUtils::DEVICE_GPU),
                       ::testing::Values(gpuConfig)),
    SetBlobOfKindTest::getTestCaseName);
