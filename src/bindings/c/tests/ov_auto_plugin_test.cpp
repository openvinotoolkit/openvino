// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_test.hpp"
#include "openvino/c/auto/auto_plugin_properties.h"

class ov_auto_plugin_test : public ov_capi_test_base {
public:
    void SetUp() override {
        ov_capi_test_base::SetUp();
    }

    void TearDown() override {
        ov_capi_test_base::TearDown();
    }
};

TEST_P(ov_auto_plugin_test, ov_core_auto_set_and_get_property_bool) {
    auto device_name = GetParam();
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    const char* key_1 = ov_property_key_intel_auto_device_bind_buffer;
    const char* key_2 = ov_property_key_intel_auto_enable_startup_fallback;
    const char* key_3 = ov_property_key_intel_auto_enable_runtime_fallback;
    const char* enable = "YES";
    const char* disable = "NO";
    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), key_1, enable));
    char* ret = nullptr;
    OV_EXPECT_OK(ov_core_get_property(core, device_name.c_str(), key_1, &ret));
    EXPECT_STREQ(enable, ret);
    ov_free(ret);
    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), key_2, enable));
    ret = nullptr;
    OV_EXPECT_OK(ov_core_get_property(core, device_name.c_str(), key_2, &ret));
    EXPECT_STREQ(enable, ret);
    ov_free(ret);
    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), key_3, enable));
    ret = nullptr;
    OV_EXPECT_OK(ov_core_get_property(core, device_name.c_str(), key_3, &ret));
    EXPECT_STREQ(enable, ret);
    ov_free(ret);
    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), key_1, disable));
    ret = nullptr;
    OV_EXPECT_OK(ov_core_get_property(core, device_name.c_str(), key_1, &ret));
    EXPECT_STREQ(disable, ret);
    ov_free(ret);
    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), key_2, disable));
    ret = nullptr;
    OV_EXPECT_OK(ov_core_get_property(core, device_name.c_str(), key_2, &ret));
    EXPECT_STREQ(disable, ret);
    ov_free(ret);
    OV_EXPECT_OK(ov_core_set_property(core, device_name.c_str(), key_3, disable));
    ret = nullptr;
    OV_EXPECT_OK(ov_core_get_property(core, device_name.c_str(), key_3, &ret));
    EXPECT_STREQ(disable, ret);
    ov_free(ret);
    ov_core_free(core);
}

INSTANTIATE_TEST_SUITE_P(ov_auto_plugin_test_properties, ov_auto_plugin_test, ::testing::Values("AUTO"));
