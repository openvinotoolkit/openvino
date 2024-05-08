//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <iostream>
#include <openvino/runtime/core.hpp>
#include <string>
#include <string_view>
#include "common/vpu_test_env_cfg.hpp"
#include "intel_npu/utils/logger/logger.hpp"

namespace ov::test::utils {

class VpuTestTool {
public:
    const VpuTestEnvConfig& envConfig;
    const std::string DEVICE_NAME;
    ::intel_npu::Logger _log;

public:
    explicit VpuTestTool(const VpuTestEnvConfig& envCfg);

    std::string getDeviceMetric(std::string name);
};

}  // namespace ov::test::utils
