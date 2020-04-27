// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <cpp/ie_plugin_cpp.hpp>
#include <cpp/ie_cnn_net_reader.h>

#include "common_test_utils/xml_net_builder/xml_net_builder.hpp"

using namespace InferenceEngine;

#define EXPECT_NO_CRASH(_statement) \
EXPECT_EXIT(_statement; exit(0), testing::ExitedWithCode(0), "")

void release_order_test(std::vector<int> order, const std::string & deviceName) {
    SizeVector dims {1,3,3,3};
    std::map<std::string, std::string> attr {
            {"power", "1"},
            {"scale", "-1"},
            {"shift", "0"}
    };

    auto model = CommonTestUtils::V2NetBuilder::buildNetworkWithOneInput("RNN_Net", dims, "FP32")
            .addLayer("Power", "FP32", &attr, {{dims}, {dims}})
            .finish();

    CNNNetwork net;

    {
        Core reader;
        net = reader.ReadNetwork(model, Blob::CPtr());
    }

    Core core;
    auto exe_net = core.LoadNetwork(net, deviceName);
    auto request = exe_net.CreateInferRequest();

    auto release = [&] (int i) {
        switch (i) {
            case 0: core = Core{}; break;
            case 1: exe_net = {}; break;
            case 2: request = {}; break;
            default: break;
        }
    };

    for (auto i : order)
        release(i);
}

class CPP_HoldersTests : public ::testing::TestWithParam<std::tuple<std::vector<int>, std::string>> {
public:
    void SetUp() override {
        order      = std::get<0>(GetParam());
        deviceName = std::get<1>(GetParam());

        deathTestStyle = ::testing::GTEST_FLAG(death_test_style);
        if (deathTestStyle == "fast" && deviceName == "MYRIAD") {
            // Default death test mode "fast" must be used in single-threaded context only.
            // "MyriadBehaviorTests" links "XLink" library that statically initializes "libusb".
            // Which in turn creates a thread.
            ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
        }
    }

    void TearDown() override {
        ::testing::GTEST_FLAG(death_test_style) = deathTestStyle;
    }

    std::string deviceName;
    std::vector<int> order;

private:
    std::string deathTestStyle;
};

TEST_P(CPP_HoldersTests, Orders) {
    // Test failed if crash happens
    EXPECT_NO_CRASH(release_order_test(order, deviceName));
}
