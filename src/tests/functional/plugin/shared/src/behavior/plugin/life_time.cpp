// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>

#include <ngraph_functions/subgraph_builders.hpp>
#include <base/behavior_test_utils.hpp>
#include "behavior/plugin/life_time.hpp"

#ifndef _WIN32
    #include <signal.h>
    #include <setjmp.h>
#endif

namespace BehaviorTestsDefinitions {

#ifndef _WIN32
    static jmp_buf env;
#endif

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
        function = ngraph::builder::subgraph::makeConvPoolRelu();

#ifndef _WIN32
        // configure handling of crash
        auto crashHandler = [](int errCode) {
            std::cerr << "Unexpected application crash with code: " << errCode << std::endl;
            siglongjmp(env, 1);
        };
        struct sigaction act;
        act.sa_handler = crashHandler;
        sigemptyset(&act.sa_mask);
        act.sa_flags = 0;
        sigaction(SIGSEGV, &act, 0);
#endif
    }

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
#ifdef _WIN32
        EXPECT_NO_THROW(release_order_test(order, targetDevice, function));
#else
        if (sigsetjmp(env, 1) == 0) {
            release_order_test(order, targetDevice, function);
        } else {
            IE_THROW() << "Crash happens";
        }
#endif
    }

    TEST_P(HoldersTestImportNetwork, Orders) {
        // Test failed if crash happens
#ifdef _WIN32
        EXPECT_NO_THROW(release_order_test(order, targetDevice, function));
#else
        if (sigsetjmp(env, 1) == 0) {
            release_order_test(order, targetDevice, function);
        } else {
            IE_THROW() << "Crash happens";
        }
#endif
    }

    std::string HoldersTestOnImportedNetwork::getTestCaseName(testing::TestParamInfo<std::string> obj) {
        return "targetDevice=" + obj.param;
    }

    void HoldersTestOnImportedNetwork::SetUp() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        targetDevice = this->GetParam();
        function = ngraph::builder::subgraph::makeConvPoolRelu();
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
