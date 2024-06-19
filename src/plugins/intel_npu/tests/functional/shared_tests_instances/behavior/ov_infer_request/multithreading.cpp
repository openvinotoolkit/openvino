// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/multithreading.hpp"
#include "common/utils.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "intel_npu/al/config/common.hpp"
#include "overload/ov_infer_request/multithreading.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> configs = {{}};

const std::vector<ov::AnyMap> multiConfigs = {
        {ov::device::priorities(ov::test::utils::DEVICE_NPU),
        ov::device::properties(ov::test::utils::DEVICE_NPU, {})}};

const std::vector<ov::AnyMap> autoConfigs = {
        {ov::device::priorities(ov::test::utils::DEVICE_NPU),
        ov::device::properties(ov::test::utils::DEVICE_NPU, {})}};

INSTANTIATE_TEST_SUITE_P(backwardDrvComp_smoke_BehaviorTests, OVInferRequestMultithreadingTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(backwardDrvComp_smoke_Multi_BehaviorTests, OVInferRequestMultithreadingTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(backwardDrvComp_smoke_Auto_BehaviorTests, OVInferRequestMultithreadingTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(backwardDrvComp_smoke_BehaviorTests, OVInferRequestMultithreadingTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(backwardDrvComp_smoke_Multi_BehaviorTests, OVInferRequestMultithreadingTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multiConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(backwardDrvComp_smoke_Auto_BehaviorTests, OVInferRequestMultithreadingTestsNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoConfigs)),
                         InferRequestParamsAnyMapTestName::getTestCaseName);

}  // namespace
