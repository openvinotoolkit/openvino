// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gtest/gtest.h"

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/common_utils.hpp"

#include "functional_test_utils/layer_test_utils/summary.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"

namespace ov {
namespace test {
namespace subgraph {

using OpImplParams = std::tuple<
        std::pair<ov::DiscreteTypeInfo, std::shared_ptr<ov::Function>>,       // Function to check
        std::string,                         // Target Device
        std::map<std::string, std::string>>; // Plugin Config

class OpImplCheckTest : public testing::WithParamInterface<OpImplParams>,
                        public CommonTestUtils::TestsCommon {
protected:
    LayerTestsUtils::Summary& summary = LayerTestsUtils::Summary::getInstance();
    std::shared_ptr<ov::runtime::Core> core = ov::test::utils::PluginCache::get().core();
    std::shared_ptr<ov::Function> function;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;

public:
    void run() {
        if (!function) {
            throw std::runtime_error("Function is empty!");
        }
        auto crashHandler = [](int errCode) {
            auto& s = LayerTestsUtils::Summary::getInstance();
            s.saveReport();
            std::cerr << "Unexpected application crash with code: " << errCode << std::endl;
            std::abort();
        };
        signal(SIGSEGV, crashHandler);

        summary.setDeviceName(targetDevice);
        try {
            auto executableNetwork = core->compile_model(function, targetDevice, configuration);
            summary.updateOPsImplStatus(function, true);
        } catch (...) {
            summary.updateOPsImplStatus(function, false);
        }
    }

    void SetUp() override {
       std::pair<ov::DiscreteTypeInfo, std::shared_ptr<ov::Function>> funcInfo;
       std::tie(funcInfo, targetDevice, configuration) = this->GetParam();
       function = funcInfo.second;
    }

    static std::string getTestCaseName(const testing::TestParamInfo<OpImplParams> &obj) {
        std::pair<ov::DiscreteTypeInfo, std::shared_ptr<ov::Function>> funcInfo;
        std::string targetDevice;
        std::map<std::string, std::string> config;
        std::tie(funcInfo, targetDevice, config) = obj.param;

        std::ostringstream result;
        std::string friendlyName = funcInfo.second ? funcInfo.second->get_friendly_name() : std::string(funcInfo.first.name + funcInfo.first.version);
                result << "Function=" << friendlyName << "_";
        result << "Device=" << targetDevice << "_";
        result << "Config=(";
        for (const auto& configItem : config) {
            result << configItem.first << "=" << configItem.second << "_";
        }
        result << ")";
        return result.str();
    }
};

TEST_P(OpImplCheckTest, hell) {
    run();
}

}   // namespace subgraph
}   // namespace test
}   // namespace ov