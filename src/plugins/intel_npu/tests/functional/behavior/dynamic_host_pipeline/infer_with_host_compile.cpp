// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_with_host_compile.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"

const std::vector<std::string> devices = {"NPU.4000", "NPU.5010"};

const std::vector<ov::AnyMap> configs = {
    {{"NPU_COMPILER_TYPE", "PLUGIN"},
     {"NPU_COMPILATION_MODE", "HostCompile"},
     {"NPU_CREATE_EXECUTOR", "0"},
     // After HostCompile default params were changed to a more performant configuration, these tests fail
     // under the new defaults and need investigation before they can be re-enabled with the new configuration
     // Untill then set old defaults explicitly to keep the tests running.
     // Track: E#218923
     {"NPU_COMPILATION_MODE_PARAMS", "dynamic-dim-alignment=false enable-auto-unrolling=false"}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferWithHostCompileTests,
                         ::testing::Combine(::testing::ValuesIn(devices), ::testing::ValuesIn(configs)),
                         ov::test::utils::appendPlatformTypeTestName<InferWithHostCompileTests>);
