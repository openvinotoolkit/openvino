// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_infer_request/batched_tensors.hpp"
#include <cctype>
#include "common/utils.hpp"
#include "common/npu_test_env_cfg.hpp"

using namespace ov::test::behavior;

namespace {

INSTANTIATE_TEST_SUITE_P(compatibility_smoke_BehaviorTests, OVInferRequestBatchedTests,
                         ::testing::Values(ov::test::utils::DEVICE_NPU),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestBatchedTests>);
}  // namespace
