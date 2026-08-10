// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dx12_remote_run.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/options.hpp"

#ifdef _WIN32

#    include <utility>
#    include <vector>

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> remoteConfigs = {{ov::log::level(ov::log::Level::WARNING)}};

// C-190157
const std::vector<ov::AnyMap> dynamicRemoteConfigs = {
    {{"NPU_COMPILER_TYPE", "PLUGIN"}, {"NPU_COMPILATION_MODE", "DefaultHW"}}};

}  // namespace

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         DX12RemoteRunTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(remoteConfigs)),
                         DX12RemoteRunTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_BehaviorTests,
    DX12RemoteRunDynamicTests,
    ::testing::Combine(::testing::Values(DX12RemoteRunDynamicTests::getFunction()),
                       ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                           {{1, 10, 12}, {1, 10, 12}},
                           {{1, 18, 15}, {1, 18, 15}}}),
                       ::testing::Values(ov::test::utils::DEVICE_NPU),
                       ::testing::ValuesIn(dynamicRemoteConfigs)),
    ov::test::utils::appendPlatformTypeTestName<OVInferRequestDynamicTests>);

#endif
