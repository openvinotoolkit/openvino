// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/runtime/core.hpp"

class ProxyPluginTests : public ::testing::Test {
public:
    ov::Core core;
    void SetUp() override {
        core.register_plugin(std::string("mock_abc_plugin") + IE_BUILD_POSTFIX, "ABC");
        core.register_plugin(std::string("mock_bde_plugin") + IE_BUILD_POSTFIX, "BDE");
    }
};

TEST_F(ProxyPluginTests, get_available_devices) {
    auto available_devices = core.get_available_devices();
    for (const auto& dev : available_devices) {
        std::cout << dev << std::endl;
    }
}
