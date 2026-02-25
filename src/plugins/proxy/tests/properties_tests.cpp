// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/runtime/properties.hpp"
#include "proxy_tests.hpp"

using namespace ov::proxy::tests;

namespace {
std::string get_string_value(const ov::Any& value) {
    if (value.empty()) {
        return "Empty";
    } else {
        return value.as<std::string>();
    }
}
}  // namespace

TEST_F(ProxyTests, get_property_on_default_uninit_device) {
    const std::string dev_name = "MOCK";
    EXPECT_EQ(0, core.get_property(dev_name, ov::num_streams));
    core.set_property(dev_name, ov::num_streams(2));
    EXPECT_EQ(2, core.get_property(dev_name, ov::num_streams));
}

TEST_F(ProxyTests, set_property_for_fallback_device) {
    const std::string dev_name = "MOCK.1";
    EXPECT_EQ(0, core.get_property(dev_name, ov::num_streams));
    core.set_property(dev_name, ov::num_streams(2));
    EXPECT_EQ(2, core.get_property(dev_name, ov::num_streams));
    core.set_property(dev_name, ov::device::properties("BDE", ov::enable_profiling(true)));
    EXPECT_EQ(false, core.get_property(dev_name, ov::enable_profiling));
}

TEST_F(ProxyTests, set_property_for_primary_device) {
    const std::string dev_name = "MOCK.1";
    core.set_property(dev_name, ov::device::properties("ABC", ov::enable_profiling(true)));
    EXPECT_EQ(true, core.get_property(dev_name, ov::enable_profiling));
}

TEST_F(ProxyTests, set_property_for_primary_device_full_name) {
    const std::string dev_name = "MOCK.1";
    core.set_property(dev_name, ov::device::properties("ABC.abc_b", ov::enable_profiling(true)));
    EXPECT_EQ(true, core.get_property(dev_name, ov::enable_profiling));
}

TEST_F(ProxyTests, get_property_on_default_device) {
    const std::string dev_name = "MOCK";
    auto supported_properties = core.get_property(dev_name, ov::supported_properties);
    EXPECT_EQ(12, supported_properties.size());
    size_t mutable_pr(0), immutable_pr(0);
    for (auto&& property : supported_properties) {
        property.is_mutable() ? mutable_pr++ : immutable_pr++;
        if (property == ov::num_streams) {
            EXPECT_EQ("0", get_string_value(core.get_property(dev_name, property)));
            core.set_property(dev_name, ov::num_streams(2));
            EXPECT_TRUE(core.get_property(dev_name, property).is<int32_t>());
            EXPECT_EQ("2", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::uuid) {
            EXPECT_EQ("000102030405060708090a0b0c0d0e0f", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::priorities) {
            auto value = core.get_property(dev_name, property).as<std::vector<std::string>>();
            ASSERT_EQ(value.size(), 2);
            EXPECT_EQ(value[0], "ABC");
            EXPECT_EQ(value[1], "BDE");
        } else {
            EXPECT_NO_THROW(core.get_property(dev_name, property));
        }
    }
    EXPECT_EQ(6, immutable_pr);
    EXPECT_EQ(6, mutable_pr);
}

TEST_F(ProxyTests, get_property_on_mixed_device) {
    const std::string dev_name = "MOCK.1";
    auto supported_properties = core.get_property(dev_name, ov::supported_properties);
    EXPECT_EQ(12, supported_properties.size());
    size_t mutable_pr(0), immutable_pr(0);
    for (auto&& property : supported_properties) {
        property.is_mutable() ? mutable_pr++ : immutable_pr++;
        if (property == ov::num_streams) {
            EXPECT_EQ("0", get_string_value(core.get_property(dev_name, property)));
            core.set_property(dev_name, ov::num_streams(2));
            EXPECT_TRUE(core.get_property(dev_name, property).is<int32_t>());
            EXPECT_EQ("2", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::uuid) {
            EXPECT_EQ("00020406080a0c0e10121416181a1c1e", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::priorities) {
            auto value = core.get_property(dev_name, property).as<std::vector<std::string>>();
            ASSERT_EQ(value.size(), 2);
            EXPECT_EQ(value[0], "ABC");
            EXPECT_EQ(value[1], "BDE");
        } else {
            core.get_property(dev_name, property);
        }
    }
    EXPECT_EQ(6, immutable_pr);
    EXPECT_EQ(6, mutable_pr);
}

TEST_F(ProxyTests, get_property_on_specified_device) {
    const std::string dev_name = "MOCK.3";
    auto supported_properties = core.get_property(dev_name, ov::supported_properties);
    EXPECT_EQ(8, supported_properties.size());
    size_t mutable_pr(0), immutable_pr(0);
    for (auto&& property : supported_properties) {
        property.is_mutable() ? mutable_pr++ : immutable_pr++;
        if (property == ov::enable_profiling) {
            EXPECT_EQ("NO", get_string_value(core.get_property(dev_name, property)));
            core.set_property(dev_name, ov::enable_profiling(true));
            EXPECT_TRUE(core.get_property(dev_name, property).is<bool>());
            EXPECT_EQ("YES", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::uuid) {
            EXPECT_EQ("0004080c1014181c2024282c3034383c", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::priorities) {
            auto value = core.get_property(dev_name, property).as<std::vector<std::string>>();
            ASSERT_EQ(value.size(), 1);
            EXPECT_EQ(value[0], "BDE");
        } else {
            EXPECT_NO_THROW(core.get_property(dev_name, property));
        }
    }
    EXPECT_EQ(5, immutable_pr);
    EXPECT_EQ(3, mutable_pr);
}

TEST_F(ProxyTests, get_property_for_changed_default_device) {
    const std::string dev_name = "MOCK";
    core.set_property(dev_name, ov::device::id(3));
    auto supported_properties = core.get_property(dev_name, ov::supported_properties);
    EXPECT_EQ(8, supported_properties.size());
    size_t mutable_pr(0), immutable_pr(0);
    for (auto&& property : supported_properties) {
        property.is_mutable() ? mutable_pr++ : immutable_pr++;
        if (property == ov::enable_profiling) {
            EXPECT_EQ("NO", get_string_value(core.get_property(dev_name, property)));
            core.set_property(dev_name, ov::enable_profiling(true));
            EXPECT_TRUE(core.get_property(dev_name, property).is<bool>());
            EXPECT_EQ("YES", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::uuid) {
            EXPECT_EQ("0004080c1014181c2024282c3034383c", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::priorities) {
            auto value = core.get_property(dev_name, property).as<std::vector<std::string>>();
            ASSERT_EQ(value.size(), 1);
            EXPECT_EQ(value[0], "BDE");
        } else {
            EXPECT_NO_THROW(core.get_property(dev_name, property));
        }
    }
    EXPECT_EQ(5, immutable_pr);
    EXPECT_EQ(3, mutable_pr);
}

TEST_F(ProxyTests, get_property_on_loaded_default_uninit_device) {
    core.get_available_devices();
    const std::string dev_name = "MOCK";
    EXPECT_EQ(0, core.get_property(dev_name, ov::num_streams));
    core.set_property(dev_name, ov::num_streams(2));
    EXPECT_EQ(2, core.get_property(dev_name, ov::num_streams));
}

TEST_F(ProxyTests, set_property_for_loaded_fallback_device) {
    core.get_available_devices();
    const std::string dev_name = "MOCK.1";
    EXPECT_EQ(0, core.get_property(dev_name, ov::num_streams));
    core.set_property(dev_name, ov::num_streams(2));
    EXPECT_EQ(2, core.get_property(dev_name, ov::num_streams));
    core.set_property(dev_name, ov::device::properties("BDE", ov::enable_profiling(true)));
    EXPECT_EQ(false, core.get_property(dev_name, ov::enable_profiling));
}

TEST_F(ProxyTests, set_cache_dir_for_loaded_fallback_device) {
    core.get_available_devices();
    const std::string dev_name = "MOCK.1";
    core.set_property(dev_name, ov::cache_dir("test_cache"));
    auto model = create_model_with_subtract();
    auto compiled_model = core.compile_model(model, "MOCK.1", ov::cache_dir("test_cache"));
    auto infer_request = compiled_model.create_infer_request();
    auto input_tensor = create_and_fill_tensor(model->input().get_element_type(), model->input().get_shape());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
}

TEST_F(ProxyTests, set_property_for_loaded_primary_device) {
    core.get_available_devices();
    const std::string dev_name = "MOCK.1";
    core.set_property(dev_name, ov::device::properties("ABC", ov::enable_profiling(true)));
    EXPECT_EQ(true, core.get_property(dev_name, ov::enable_profiling));
}

TEST_F(ProxyTests, set_property_for_loaded_primary_device_full_name) {
    core.get_available_devices();
    const std::string dev_name = "MOCK.1";
    core.set_property(dev_name, ov::device::properties("ABC.abc_b", ov::enable_profiling(true)));
    EXPECT_EQ(true, core.get_property(dev_name, ov::enable_profiling));
}

TEST_F(ProxyTests, get_property_on_loaded_default_device) {
    core.get_available_devices();
    const std::string dev_name = "MOCK";
    auto supported_properties = core.get_property(dev_name, ov::supported_properties);
    EXPECT_EQ(12, supported_properties.size());
    size_t mutable_pr(0), immutable_pr(0);
    for (auto&& property : supported_properties) {
        property.is_mutable() ? mutable_pr++ : immutable_pr++;
        if (property == ov::num_streams) {
            EXPECT_EQ("0", get_string_value(core.get_property(dev_name, property)));
            core.set_property(dev_name, ov::num_streams(2));
            EXPECT_TRUE(core.get_property(dev_name, property).is<int32_t>());
            EXPECT_EQ("2", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::uuid) {
            EXPECT_EQ("000102030405060708090a0b0c0d0e0f", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::priorities) {
            auto value = core.get_property(dev_name, property).as<std::vector<std::string>>();
            ASSERT_EQ(value.size(), 2);
            EXPECT_EQ(value[0], "ABC");
            EXPECT_EQ(value[1], "BDE");
        } else {
            EXPECT_NO_THROW(core.get_property(dev_name, property));
        }
    }
    EXPECT_EQ(6, immutable_pr);
    EXPECT_EQ(6, mutable_pr);
}

TEST_F(ProxyTests, get_property_loaded_on_mixed_device) {
    core.get_available_devices();
    const std::string dev_name = "MOCK.1";
    auto supported_properties = core.get_property(dev_name, ov::supported_properties);
    EXPECT_EQ(12, supported_properties.size());
    size_t mutable_pr(0), immutable_pr(0);
    for (auto&& property : supported_properties) {
        property.is_mutable() ? mutable_pr++ : immutable_pr++;
        if (property == ov::num_streams) {
            EXPECT_EQ("0", get_string_value(core.get_property(dev_name, property)));
            core.set_property(dev_name, ov::num_streams(2));
            EXPECT_TRUE(core.get_property(dev_name, property).is<int32_t>());
            EXPECT_EQ("2", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::uuid) {
            EXPECT_EQ("00020406080a0c0e10121416181a1c1e", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::priorities) {
            auto value = core.get_property(dev_name, property).as<std::vector<std::string>>();
            ASSERT_EQ(value.size(), 2);
            EXPECT_EQ(value[0], "ABC");
            EXPECT_EQ(value[1], "BDE");
        } else {
            core.get_property(dev_name, property);
        }
    }
    EXPECT_EQ(6, immutable_pr);
    EXPECT_EQ(6, mutable_pr);
}

TEST_F(ProxyTests, get_property_loaded_on_specified_device) {
    core.get_available_devices();
    const std::string dev_name = "MOCK.3";
    auto supported_properties = core.get_property(dev_name, ov::supported_properties);
    EXPECT_EQ(8, supported_properties.size());
    size_t mutable_pr(0), immutable_pr(0);
    for (auto&& property : supported_properties) {
        property.is_mutable() ? mutable_pr++ : immutable_pr++;
        if (property == ov::enable_profiling) {
            EXPECT_EQ("NO", get_string_value(core.get_property(dev_name, property)));
            core.set_property(dev_name, ov::enable_profiling(true));
            EXPECT_TRUE(core.get_property(dev_name, property).is<bool>());
            EXPECT_EQ("YES", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::uuid) {
            EXPECT_EQ("0004080c1014181c2024282c3034383c", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::priorities) {
            auto value = core.get_property(dev_name, property).as<std::vector<std::string>>();
            ASSERT_EQ(value.size(), 1);
            EXPECT_EQ(value[0], "BDE");
        } else {
            EXPECT_NO_THROW(core.get_property(dev_name, property));
        }
    }
    EXPECT_EQ(5, immutable_pr);
    EXPECT_EQ(3, mutable_pr);
}

TEST_F(ProxyTests, get_property_for_loaded_changed_default_device) {
    core.get_available_devices();
    const std::string dev_name = "MOCK";
    core.set_property(dev_name, ov::device::id(3));
    auto supported_properties = core.get_property(dev_name, ov::supported_properties);
    EXPECT_EQ(8, supported_properties.size());
    size_t mutable_pr(0), immutable_pr(0);
    for (auto&& property : supported_properties) {
        property.is_mutable() ? mutable_pr++ : immutable_pr++;
        if (property == ov::enable_profiling) {
            EXPECT_EQ("NO", get_string_value(core.get_property(dev_name, property)));
            core.set_property(dev_name, ov::enable_profiling(true));
            EXPECT_TRUE(core.get_property(dev_name, property).is<bool>());
            EXPECT_EQ("YES", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::uuid) {
            EXPECT_EQ("0004080c1014181c2024282c3034383c", get_string_value(core.get_property(dev_name, property)));
        } else if (property == ov::device::priorities) {
            auto value = core.get_property(dev_name, property).as<std::vector<std::string>>();
            ASSERT_EQ(value.size(), 1);
            EXPECT_EQ(value[0], "BDE");
        } else {
            EXPECT_NO_THROW(core.get_property(dev_name, property));
        }
    }
    EXPECT_EQ(5, immutable_pr);
    EXPECT_EQ(3, mutable_pr);
}

#ifdef ENABLE_AUTO_BATCH

#    ifdef HETERO_ENABLED

TEST_F(ProxyTests, set_cache_dir_for_auto_batch_hetero_fallback_device) {
    core.get_available_devices();
    const std::string dev_name = "MOCK.1";
    core.set_property(dev_name, ov::cache_dir("test_cache"));
    auto model = create_model_with_add();
    auto compiled_model = core.compile_model(model,
                                             "MOCK.1",
                                             ov::cache_dir("test_cache"),
                                             ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
    auto infer_request = compiled_model.create_infer_request();
    auto input_tensor = create_and_fill_tensor(model->input().get_element_type(), model->input().get_shape());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
}

#    endif

TEST_F(ProxyTests, set_cache_dir_for_auto_batch_main_fallback_device) {
    core.get_available_devices();
    const std::string dev_name = "MOCK.1";
    core.set_property(dev_name, ov::cache_dir("test_cache"));
    auto model = create_model_with_add();
    auto compiled_model = core.compile_model(model,
                                             "MOCK.0",
                                             ov::cache_dir("test_cache"),
                                             ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
    auto infer_request = compiled_model.create_infer_request();
    auto input_tensor = create_and_fill_tensor(model->input().get_element_type(), model->input().get_shape());
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
}

#endif
