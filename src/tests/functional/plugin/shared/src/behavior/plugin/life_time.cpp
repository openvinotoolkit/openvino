// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>

#include <ngraph_functions/subgraph_builders.hpp>
#include <base/behavior_test_utils.hpp>
#include "behavior/plugin/life_time.hpp"

namespace BehaviorTestsDefinitions {
    std::string HoldersTest::getTestCaseName(testing::TestParamInfo<HoldersParams> obj) {
        std::string targetDevice;
        std::vector<int> order;
        std::tie(targetDevice, order) = obj.param;
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        if (!order.empty()) {
            std::string objects[] = { "core", "exec-net", "request", "state" };
            for (auto &Item : order) {
                result << objects[Item] << "_";
            }
        }
        return result.str();
    }

    void HoldersTest::SetUp() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::tie(targetDevice, order) = this->GetParam();
        deathTestStyle = ::testing::GTEST_FLAG(death_test_style);
        if (deathTestStyle == "fast") {
            ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
        }
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void HoldersTest::TearDown() {
        ::testing::GTEST_FLAG(death_test_style) = deathTestStyle;
    }

#define EXPECT_NO_CRASH(_statement) \
    EXPECT_EXIT(_statement; exit(0), testing::ExitedWithCode(0), "")

    void release_order_test(std::vector<int> order, const std::string &deviceName,
                            std::shared_ptr<ngraph::Function> function) {
        InferenceEngine::CNNNetwork cnnNet(function);
        InferenceEngine::Core core = BehaviorTestsUtils::createIECoreWithTemplate();
        auto exe_net = core.LoadNetwork(cnnNet, deviceName);
        auto request = exe_net.CreateInferRequest();
        std::vector<InferenceEngine::VariableState> states;
        try {
            states = request.QueryState();
        } catch(...) {
            // do nothing
        }

        auto release = [&](int i) {
            switch (i) {
                case 0:
                    core = BehaviorTestsUtils::createIECoreWithTemplate();
                    break;
                case 1:
                    exe_net = {};
                    break;
                case 2:
                    request = {};
                    break;
                case 3:
                    states = {};
                    break;
                default:
                    break;
            }
        };

        for (auto i : order)
            release(i);
    }

    TEST_P(HoldersTest, Orders) {
        // Test failed if crash happens
        EXPECT_NO_CRASH(release_order_test(order, targetDevice, function));
    }

    TEST_P(HoldersTestImportNetwork, Orders) {
        // Test failed if crash happens
        EXPECT_NO_CRASH(release_order_test(order, targetDevice, function));
    }

    std::string HoldersTestOnImportedNetwork::getTestCaseName(testing::TestParamInfo<std::string> obj) {
        return "targetDevice=" + obj.param;
    }

    void HoldersTestOnImportedNetwork::SetUp() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        targetDevice = this->GetParam();
        deathTestStyle = ::testing::GTEST_FLAG(death_test_style);
        if (deathTestStyle == "fast") {
            ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
        }
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void HoldersTestOnImportedNetwork::TearDown() {
        ::testing::GTEST_FLAG(death_test_style) = deathTestStyle;
    }

    TEST_P(HoldersTestOnImportedNetwork, CreateRequestWithCoreRemoved) {
        InferenceEngine::CNNNetwork cnnNet(function);
        InferenceEngine::Core core = BehaviorTestsUtils::createIECoreWithTemplate();
        std::stringstream stream;
        {
            auto exe_net = core.LoadNetwork(cnnNet, targetDevice);
            exe_net.Export(stream);
        }
        auto exe_net = core.ImportNetwork(stream, targetDevice);
        core = InferenceEngine::Core();
        auto request = exe_net.CreateInferRequest();
    }
}  // namespace BehaviorTestsDefinitions
