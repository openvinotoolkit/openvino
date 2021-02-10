// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/set_blob_of_kind.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace BehaviorTestsDefinitions;
using namespace InferenceEngine;

const std::vector<FuncTestUtils::BlobKind> blobKinds = {
    FuncTestUtils::BlobKind::Simple,
    FuncTestUtils::BlobKind::Compound,
    FuncTestUtils::BlobKind::BatchOfSimple
};

const SetBlobOfKindConfig gpuConfig{}; //nothing special

INSTANTIATE_TEST_CASE_P(smoke_SetBlobOfKindGPU, SetBlobOfKindTest,
    ::testing::Combine(::testing::ValuesIn(blobKinds),
                       ::testing::Values(CommonTestUtils::DEVICE_GPU),
                       ::testing::Values(gpuConfig)),
    SetBlobOfKindTest::getTestCaseName);
