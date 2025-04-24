// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_test.hpp"

using test_params = std::tuple<std::string, const char*, const char*, bool>;

class ov_auto_plugin_test : public ::testing::TestWithParam<test_params> {
public:
    std::string device_name;
    const char* auto_property;
    const char* property_value;
    bool invalid_value;

public:
    void SetUp() override {
        std::tie(device_name, auto_property, property_value, invalid_value) = GetParam();
    }
};

TEST_P(ov_auto_plugin_test, ov_core_auto_set_and_get_property_bool) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);
    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), auto_property, property_value));
    char* ret = nullptr;
    if (invalid_value) {
        OV_EXPECT_NOT_OK(ov_core_get_property(core, device_name.c_str(), auto_property, &ret));
        EXPECT_STRNE(property_value, ret);
    } else {
        OV_EXPECT_OK(ov_core_get_property(core, device_name.c_str(), auto_property, &ret));
        EXPECT_STREQ(property_value, ret);
    }
    ov_free(ret);
    ov_core_free(core);
}

const std::vector<test_params> test_property_config = {
    test_params{"AUTO", ov_property_key_intel_auto_device_bind_buffer, "YES", false},
    test_params{"AUTO", ov_property_key_intel_auto_device_bind_buffer, "NO", false},
    test_params{"AUTO", ov_property_key_intel_auto_device_bind_buffer, "TEST", true},
    test_params{"AUTO", ov_property_key_intel_auto_enable_startup_fallback, "YES", false},
    test_params{"AUTO", ov_property_key_intel_auto_enable_startup_fallback, "NO", false},
    test_params{"AUTO", ov_property_key_intel_auto_enable_startup_fallback, "TEST", true},
    test_params{"AUTO", ov_property_key_intel_auto_enable_runtime_fallback, "YES", false},
    test_params{"AUTO", ov_property_key_intel_auto_enable_runtime_fallback, "NO", false},
    test_params{"AUTO", ov_property_key_intel_auto_enable_runtime_fallback, "TEST", true},
};

INSTANTIATE_TEST_SUITE_P(ov_auto_plugin_test_properties,
                         ov_auto_plugin_test,
                         ::testing::ValuesIn(test_property_config));
