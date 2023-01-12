// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/core.hpp"

#include <gtest/gtest.h>

#include <fstream>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/util/file_util.hpp"

TEST(CoreTests, ThrowOnRegisterPluginTwice) {
    ov::Core core;
    core.register_plugin("test_plugin", "TEST_DEVICE");
    OV_EXPECT_THROW(core.register_plugin("test_plugin", "TEST_DEVICE"),
                    ov::Exception,
                    ::testing::HasSubstr("Device with \"TEST_DEVICE\"  is already registered in the OpenVINO Runtime"));
}

TEST(CoreTests, ThrowOnRegisterPluginsTwice) {
    ov::Core core;

    auto getPluginXml = [&]() -> std::string {
        std::string pluginsXML = "test_plugins.xml";
        std::ofstream file(pluginsXML);
        file << "<ie><plugins><plugin location=\"libtest_plugin.so\" name=\"TEST_DEVICE\"></plugin></plugins></ie>";
        file.flush();
        file.close();
        return pluginsXML;
    };

    core.register_plugins(getPluginXml());
    OV_EXPECT_THROW(core.register_plugins(getPluginXml()),
                    ov::Exception,
                    ::testing::HasSubstr("Device with \"TEST_DEVICE\"  is already registered in the OpenVINO Runtime"));
}