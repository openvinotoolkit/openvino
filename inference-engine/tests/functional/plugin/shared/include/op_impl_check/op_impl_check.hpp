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
        std::shared_ptr<ov::Function>,       // Function to check
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
       std::tie(function, targetDevice, configuration) = this->GetParam();
    }

    static std::string getTestCaseName(const testing::TestParamInfo<OpImplParams> &obj) {
        std::shared_ptr<ov::Function> function;
        std::string targetDevice;
        std::map<std::string, std::string> config;
        std::tie(function, targetDevice, config) = obj.param;

        std::ostringstream result;
        result << "Function=" << function->get_friendly_name() << "_";
        result << "Device=" << targetDevice << "_";
        result << "Config=(";
        for (const auto& configItem : config) {
            result << configItem.first << "=" << configItem.second << "_";
        }
        result << ")";
        return result.str();
    }
};

TEST_P(OpImplCheckTest, ) {
    run();
}

}   // namespace subgraph
}   // namespace test
}   // namespace ov