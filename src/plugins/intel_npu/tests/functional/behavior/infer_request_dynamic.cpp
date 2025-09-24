//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request_dynamic.hpp"

#include <vector>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/options.hpp"

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> config = {{{"NPU_COMPILATION_MODE", "ReferenceSW"}}};

INSTANTIATE_TEST_SUITE_P(
    smoke_BehaviorTests,
    InferRequestDynamicTests,
    ::testing::Combine(::testing::Values(InferRequestDynamicTests::getFunction()),
                       ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                           {{1, 10, 18}, {6, 12, 15}}, {{1, 2, 14}, {5, 11, 18}}}),
                       ::testing::Values(ov::test::utils::DEVICE_NPU),
                       ::testing::ValuesIn(config)),
    ov::test::utils::appendPlatformTypeTestName<OVInferRequestDynamicTests>);
