// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>
#include <string>
#include <vector>

#include "common/functions.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace {

class TestPluginCompilerCompilationNPU : public ov::test::behavior::OVPluginTestBase,
                                         public testing::WithParamInterface<std::tuple<std::string, ov::AnyMap>> {
public:
    void SetUp() override {
        std::tie(target_device, configuration) = GetParam();
        OVPluginTestBase::SetUp();
    }

    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<std::string, ov::AnyMap>>& obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        if (!configuration.empty()) {
            using namespace ov::test::utils;
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }

protected:
    ov::AnyMap configuration;
    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
};

TEST_P(TestPluginCompilerCompilationNPU, compileWithDefaultCompilerConfig) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        const auto& model = buildSingleLayerSoftMaxNetwork();
        ov::CompiledModel compiledModel;
        OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, configuration));
        std::stringstream modelStream;
        OV_ASSERT_NO_THROW(compiledModel.export_model(modelStream));
    }
}

}  // namespace
