// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/batched_tensors.hpp"

using namespace ov::test::behavior;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_GPU, OVInferRequestBatchedTests,
                         ::testing::Values(CommonTestUtils::DEVICE_GPU),
                         OVInferRequestBatchedTests::getTestCaseName);

}  // namespace
