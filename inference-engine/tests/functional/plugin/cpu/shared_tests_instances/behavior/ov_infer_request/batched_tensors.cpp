// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/batched_tensors.hpp"

using namespace ov::test::behavior;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests_1, OVInferRequestBatchedTests,
                         ::testing::Values(CommonTestUtils::DEVICE_CPU),
                         OVInferRequestBatchedTests::getTestCaseName);

}  // namespace
