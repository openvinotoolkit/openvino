// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/runtime/core.hpp"

class ProxyPluginTests : public ::testing::Test {
public:
    ov::Core core;
    void SetUp() override {
        // TODO: Remove temp plugins from core
        // core.register_plugin(std::string("mock_abc_plugin") + IE_BUILD_POSTFIX, "ABC");
        // core.register_plugin(std::string("mock_bde_plugin") + IE_BUILD_POSTFIX, "BDE");
    }
};

TEST_F(ProxyPluginTests, get_available_devices) {
    auto available_devices = core.get_available_devices();
    std::set<std::string> mock_reference_dev = {"MOCK.0", "MOCK.1", "MOCK.2", "MOCK.3", "MOCK.4"};
    for (const auto& dev : available_devices) {
        if (mock_reference_dev.find(dev) != mock_reference_dev.end()) {
            mock_reference_dev.erase(dev);
        }
    }
    // All devices should be found
    EXPECT_TRUE(mock_reference_dev.empty());
}
