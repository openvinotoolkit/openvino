// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <tuple>
#include <gtest/gtest.h>
#include <ngraph/function.hpp>
#include <ie_plugin_config.hpp>
#include <ngraph/function.hpp>
#include <common_test_utils/test_constants.hpp>
#include <cpp/ie_cnn_network.h>
#include "gtest/gtest.h"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "../../../../../ngraph_functions/include/ngraph_functions/subgraph_builders.hpp"
#include <ie_core.hpp>

typedef std::tuple<
        std::string,         // Target device name
        std::vector<int>>    // Order
HoldersParams;

class HoldersTest : public CommonTestUtils::TestsCommon,
                   public ::testing::WithParamInterface<HoldersParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<HoldersParams> obj) {
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

    void SetUp() override {
        std::tie(targetDevice, order) = this->GetParam();
        deathTestStyle = ::testing::GTEST_FLAG(death_test_style);
        if (deathTestStyle == "fast" && targetDevice == CommonTestUtils::DEVICE_MYRIAD) {
            // Default death test mode "fast" must be used in single-threaded context only.
            // "MyriadBehaviorTests" links "XLink" library that statically initializes "libusb".
            // Which in turn creates a thread.
            ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
        }
          function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void TearDown() override {
        ::testing::GTEST_FLAG(death_test_style) = deathTestStyle;
        function.reset();
    }

    std::string deathTestStyle;
    std::vector<int> order;
    std::shared_ptr<ngraph::Function> function;
    std::string targetDevice;
};

#define EXPECT_NO_CRASH(_statement) \
EXPECT_EXIT(_statement; exit(0), testing::ExitedWithCode(0), "")

void release_order_test(std::vector<int> order, const std::string & deviceName,  std::shared_ptr<ngraph::Function> function) {
    InferenceEngine::CNNNetwork cnnNet(function);
    InferenceEngine::Core core;
    auto exe_net = core.LoadNetwork(cnnNet, deviceName);
    auto request = exe_net.CreateInferRequest();

    auto release = [&] (int i) {
        switch (i) {
            case 0: core = InferenceEngine::Core{}; break;
            case 1: exe_net = {}; break;
            case 2: request = {}; break;
            default: break;
        }
    };

    for (auto i : order)
        release(i);
}

TEST_P(HoldersTest, Orders) {
    // Test failed if crash happens
    EXPECT_NO_CRASH(release_order_test(order, targetDevice, function));
}