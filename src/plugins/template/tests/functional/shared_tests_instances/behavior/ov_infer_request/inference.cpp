// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/inference.hpp"

#include <vector>

namespace {

using namespace ov::test::behavior;
using namespace ov;

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferRequestInferenceTests,
                         ::testing::Combine(::testing::Values(tensor_roi::roi_nchw(), tensor_roi::roi_1d()),
                                            ::testing::Values(ov::test::utils::DEVICE_TEMPLATE)),
                         OVInferRequestInferenceTests::getTestCaseName);

}  // namespace
