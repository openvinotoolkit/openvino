// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <ngraph_functions/subgraph_builders.hpp>
#include "behavior/cpp_holders.hpp"

namespace BehaviorTestsUtils {
    std::string HoldersTest::getTestCaseName(testing::TestParamInfo<HoldersParams> obj) {
        std::string targetDevice;
        std::vector<int> order;
        std::tie(targetDevice, order) = obj.param;
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        if (!order.empty()) {
            for (auto &Item : order) {
                result << "order=" << Item << "_";
            }
        }
        return result.str();
    }

    void HoldersTest::SetUp() {
        std::tie(targetDevice, order) = this->GetParam();
        deathTestStyle = ::testing::GTEST_FLAG(death_test_style);
        if ((deathTestStyle == "fast" && targetDevice == CommonTestUtils::DEVICE_MYRIAD) ||
            targetDevice == CommonTestUtils::DEVICE_GPU) {
            // Default death test mode "fast" must be used in single-threaded context only.
            // "MyriadBehaviorTests" links "XLink" library that statically initializes "libusb".
            // Which in turn creates a thread.
            ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
        }
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void HoldersTest::TearDown() {
        ::testing::GTEST_FLAG(death_test_style) = deathTestStyle;
        function.reset();
    }

#define EXPECT_NO_CRASH(_statement) \
EXPECT_EXIT(_statement; exit(0), testing::ExitedWithCode(0), "")

    void release_order_test(std::vector<int> order, const std::string &deviceName,
                            std::shared_ptr<ngraph::Function> function) {
        InferenceEngine::CNNNetwork cnnNet(function);
        InferenceEngine::Core core;
        auto exe_net = core.LoadNetwork(cnnNet, deviceName);
        auto request = exe_net.CreateInferRequest();

        auto release = [&](int i) {
            switch (i) {
                case 0:
                    core = InferenceEngine::Core{};
                    break;
                case 1:
                    exe_net = {};
                    break;
                case 2:
                    request = {};
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
}  // namespace BehaviorTestsUtils