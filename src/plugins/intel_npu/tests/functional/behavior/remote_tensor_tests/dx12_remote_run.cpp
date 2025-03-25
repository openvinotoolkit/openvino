// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/remote_tensor_tests/dx12_remote_run.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/common.hpp"

#ifdef _WIN32
#    ifdef ENABLE_DX12

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> remoteConfigs = {{ov::log::level(ov::log::Level::WARNING)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         DX12RemoteRunTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(remoteConfigs)),
                         DX12RemoteRunTests::getTestCaseName);

#    endif
#endif
