//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_test_tool.hpp"
#include <functional_test_utils/ov_plugin_cache.hpp>

#include <fstream>
#include <iostream>
#include <sstream>

namespace ov::test::utils {

VpuTestTool::VpuTestTool(const VpuTestEnvConfig& envCfg)
        : envConfig(envCfg),
          DEVICE_NAME(envConfig.IE_NPU_TESTS_DEVICE_NAME.empty() ? "NPU" : envConfig.IE_NPU_TESTS_DEVICE_NAME),
          _log("VpuTestTool", ov::log::Level::INFO) {
}

std::string VpuTestTool::getDeviceMetric(std::string name) {
    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core(DEVICE_NAME);

    return core->get_property(DEVICE_NAME, name).as<std::string>();
}

}  // namespace ov::test::utils
