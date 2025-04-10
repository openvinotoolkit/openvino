// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdlib.h>

#include "base/ov_behavior_test_utils.hpp"
#include "common/npu_test_env_cfg.hpp"

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

namespace ov {
namespace test {
namespace behavior {
class DriverCompilerAdapterExpectedThrowNPU :
        public ov::test::behavior::OVPluginTestBase,
        public testing::WithParamInterface<CompilationParams> {
protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> ov_model;

public:
    static std::string getTestCaseName(testing::TestParamInfo<CompilationParams> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }

        return result.str();
    }

    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();

        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();
        ov_model = getDefaultNGraphFunctionForTheDevice();
        ov::AnyMap params;
        for (auto&& v : configuration) {
            params.emplace(v.first, v.second);
        }
    }

    std::string getEnv(const char* name) {
        const char* ret = getenv(name);
        if (!ret)
            return std::string();
        return std::string(ret);
    }
};

TEST_P(DriverCompilerAdapterExpectedThrowNPU, CheckWrongGraphExtAndThrow) {
#if defined(NPU_PLUGIN_DEVELOPER_BUILD)
    const char* name = "ADAPTER_MANUAL_CONFIG";
    std::string env_value = getEnv(name);

#ifdef _WIN32
    _putenv_s(name, "WRONG_VERSION");
#else
    setenv(name, "WRONG_VERSION", 1);
#endif

    EXPECT_THROW(auto compiledModel = core->compile_model(ov_model, target_device, configuration);, std::exception);

#ifdef _WIN32
    _putenv_s(name, env_value.c_str());
#else
    if (!env_value.empty()) {
        setenv(name, env_value.c_str(), 1);
    } else {
        unsetenv(name);
    }
#endif
#endif
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
