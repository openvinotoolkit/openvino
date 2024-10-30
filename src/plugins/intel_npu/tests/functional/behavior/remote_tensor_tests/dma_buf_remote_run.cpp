// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/remote_tensor_tests/dma_buf_remote_run.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/common.hpp"

#ifdef __linux__
#    include <linux/version.h>
#    if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 10, 0)

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> remoteConfigs = {{ov::log::level(ov::log::Level::WARNING)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         DmaBufRemoteRunTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(remoteConfigs)),
                         DmaBufRemoteRunTests::getTestCaseName);

#    endif
#endif
