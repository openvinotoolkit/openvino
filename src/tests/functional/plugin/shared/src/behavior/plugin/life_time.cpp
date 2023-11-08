// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>

#include <ov_models/subgraph_builders.hpp>
#include <base/behavior_test_utils.hpp>
#include "behavior/plugin/life_time.hpp"

#include <setjmp.h>

namespace BehaviorTestsDefinitions {
    std::string HoldersTest::getTestCaseName(testing::TestParamInfo<HoldersParams> obj) {
        std::string target_device;
        std::vector<int> order;
        std::tie(target_device, order) = obj.param;
        std::replace(target_device.begin(), target_device.end(), ':', '.');
        std::ostringstream result;
        result << "target_device=" << target_device << "_";
        if (!order.empty()) {
            std::string objects[] = { "core", "exec.net", "request", "state" };
            for (auto &Item : order) {
                result << objects[Item] << "_";
            }
        }
        return result.str();
    }

    void HoldersTest::SetUp() {
        std::tie(target_device, order) = this->GetParam();
        APIBaseTest::SetUp();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void release_order_test(std::vector<int> order, const std::string &target_device,
                            std::shared_ptr<ngraph::Function> function) {
        InferenceEngine::CNNNetwork cnnNet(function);
        InferenceEngine::Core core = BehaviorTestsUtils::createIECoreWithTemplate();
        auto exe_net = core.LoadNetwork(cnnNet, target_device);
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
        // in case of crash jump will be made and work will be continued
        auto crashHandler = std::unique_ptr<ov::test::utils::CrashHandler>(new ov::test::utils::CrashHandler());

        // Test failed if crash happens
#ifdef _WIN32
        if (setjmp(ov::test::utils::env) == ov::test::utils::JMP_STATUS::ok) {
#else
        if (sigsetjmp(ov::test::utils::env, 1) == ov::test::utils::JMP_STATUS::ok) {
#endif
            EXPECT_NO_THROW(release_order_test(order, target_device, function));
        } else {
            IE_THROW() << "Crash happens";
        }
    }

    TEST_P(HoldersTestImportNetwork, Orders) {
        // in case of crash jump will be made and work will be continued
        auto crashHandler = std::unique_ptr<ov::test::utils::CrashHandler>(new ov::test::utils::CrashHandler());

        // Test failed if crash happens
#ifdef _WIN32
        if (setjmp(ov::test::utils::env) == ov::test::utils::JMP_STATUS::ok) {
#else
        if (sigsetjmp(ov::test::utils::env, 1) == ov::test::utils::JMP_STATUS::ok) {
#endif
            EXPECT_NO_THROW(release_order_test(order, target_device, function));
        } else {
            IE_THROW() << "Crash happens";
        }
    }

    std::string HoldersTestOnImportedNetwork::getTestCaseName(testing::TestParamInfo<std::string> obj) {
        return "target_device=" + obj.param;
    }

    void HoldersTestOnImportedNetwork::SetUp() {
        target_device = this->GetParam();
        APIBaseTest::SetUp();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
    }

    TEST_P(HoldersTestOnImportedNetwork, CreateRequestWithCoreRemoved) {
        InferenceEngine::CNNNetwork cnnNet(function);
        InferenceEngine::Core core = BehaviorTestsUtils::createIECoreWithTemplate();
        std::stringstream stream;
        {
            auto exe_net = core.LoadNetwork(cnnNet, target_device);
            exe_net.Export(stream);
        }
        auto exe_net = core.ImportNetwork(stream, target_device);
        core = InferenceEngine::Core();
        auto request = exe_net.CreateInferRequest();
    }
}  // namespace BehaviorTestsDefinitions
