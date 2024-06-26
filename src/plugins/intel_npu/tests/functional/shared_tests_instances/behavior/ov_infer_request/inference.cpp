// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/inference.hpp"
#include "common/utils.hpp"
#include "common/npu_test_env_cfg.hpp"

namespace {

using namespace ov::test::behavior;

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests, OVInferRequestInferenceTests,
                         ::testing::Combine(::testing::Values(tensor_roi::roi_nchw(), tensor_roi::roi_1d()),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestInferenceTests>);

}  // namespace
