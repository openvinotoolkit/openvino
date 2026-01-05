// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "roi_tensors_tests.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/options.hpp"

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> roiConfigs = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         RoiTensorsTestsRun,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(roiConfigs)),
                         RoiTensorsTestsRun::getTestCaseName);
