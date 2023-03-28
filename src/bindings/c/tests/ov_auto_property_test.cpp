// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_test.hpp"

using ConfigParams = std::tuple<std::string, const char*, const char*>;

class ov_auto_plugin_test : public ::testing::TestWithParam<ConfigParams> {
public:
    std::string device_name;
    const char* auto_property;
    const char* statues;

public:
    void SetUp() override {
        std::tie(device_name, auto_property, statues) = GetParam();
    }
};

TEST_P(ov_auto_plugin_test, ov_core_auto_set_and_get_property_bool) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);
    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), auto_property, statues));
    char* ret = nullptr;
    OV_EXPECT_OK(ov_core_get_property(core, device_name.c_str(), auto_property, &ret));
    EXPECT_STREQ(statues, ret);
    ov_free(ret);
    ov_core_free(core);
}

const std::vector<ConfigParams> testCtputConfigs = {
    ConfigParams{"AUTO", ov_property_key_intel_auto_device_bind_buffer, "YES"},
    ConfigParams{"AUTO", ov_property_key_intel_auto_device_bind_buffer, "NO"},
    ConfigParams{"AUTO", ov_property_key_intel_auto_enable_startup_fallback, "YES"},
    ConfigParams{"AUTO", ov_property_key_intel_auto_enable_startup_fallback, "NO"},
    ConfigParams{"AUTO", ov_property_key_intel_auto_enable_runtime_fallback, "YES"},
    ConfigParams{"AUTO", ov_property_key_intel_auto_enable_runtime_fallback, "NO"},
};

INSTANTIATE_TEST_SUITE_P(ov_auto_plugin_test_properties, ov_auto_plugin_test, ::testing::ValuesIn(testCtputConfigs));
