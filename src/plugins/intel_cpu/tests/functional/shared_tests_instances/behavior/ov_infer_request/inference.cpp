// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/inference.hpp"

namespace {

using namespace ov::test::behavior;
using namespace ov;

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestInferenceTests,
                         ::testing::Combine(
                                 ::testing::Values(tensor_roi::roi_nchw(), tensor_roi::roi_1d()),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         OVInferRequestInferenceTests::getTestCaseName);

}  // namespace
